"""
`logmelspectrogram` with the mel table handed over as a HOST matrix, which is how
Whisper calls it: `melfilters` builds the table on the host.

It is uploaded inside, and the upload is a `Mantle.Buffer` — a resource, not an
array — so `F * mag` had no method and Whisper's transcription stopped at its
first window. The result has to be the one the device-resident table gives.

The first fix unwrapped it as `storage(toback(...))` and dropped the `Buffer`,
whose finalizer retires the region: inside the full DNNKernels suite a
collection landed before the multiply was submitted and the GPU faulted on the
retired pages. The fix holds the `Buffer` across the submit; this file runs as
part of that suite, which is what reproduces it.
"""

using Test, Random
import DNNKernels, Mantle
const DK = DNNKernels

function logmelhost(be)
    dev = Mantle.todevice(be)
    @testset "log-mel with a host filter table on $(nameof(typeof(dev)))" begin
        Random.seed!(2)
        # The `Buffer`s stay bound until freed: `storage` does not keep one alive.
        ab = DK.toback(be, 0.1f0 .* randn(Float32, 16000))
        host = DK.melfilters(80, 201, 16000)
        fb = DK.toback(be, host)
        fromhost = Array(DK.logmelspectrogram(be, Mantle.storage(ab), host))
        ondevice = Array(DK.logmelspectrogram(be, Mantle.storage(ab), Mantle.storage(fb)))
        @test size(fromhost) == (80, 100)
        @test all(isfinite, fromhost)
        @test fromhost == ondevice
        Mantle.free!(ab); Mantle.free!(fb)
    end
end

let bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping log-mel"
    foreach(logmelhost, bes)
end
