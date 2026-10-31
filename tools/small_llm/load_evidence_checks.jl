# Load in the workspace root Julia environment. Loading does not select a device.
# CPU: test_portable_evidence(Mantle.Device("llvmpipe"))
# GPU: test_portable_evidence(Mantle.Device("radeon"))
include("device_finish.jl")
include("portable_evidence.jl")
include("reference/structured_evidence.jl")
include("test_portable_evidence.jl")
