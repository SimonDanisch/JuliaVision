module OIDNRunnerVideoEditorExt
using OIDNRunner, VideoEditor
const KA = OIDNRunner.KA

# One denoiser per device the frames come from: the networks run where the
# frame's planes are, so nothing goes to the host and back.
const DENOISERS = Dict{Any, OIDNDenoiser}()

function denoiserfor(backend)
    return get!(() -> OIDNDenoiser(; backend), DENOISERS, backend)
end

function __init__()
    VideoEditor.registerhdrdenoiser!(:oidn) do out, frame, image, effect
        d = denoiserfor(KA.get_backend(image))
        return OIDNRunner.denoise!(d, out, image, frame.albedo, frame.normal; clean_aux = !effect.prefilter)
    end
end

function OIDNRunner.install!(d::OIDNDenoiser)
    VideoEditor.registerhdrdenoiser!(:oidn) do out, frame, image, effect
        return OIDNRunner.denoise!(d, out, image, frame.albedo, frame.normal; clean_aux = !effect.prefilter)
    end
    return d
end
end
