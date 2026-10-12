module OIDNRunnerVideoEditorExt
using OIDNRunner, VideoEditor
using ColorTypes: RGB
const DEFAULT = Ref{Any}(nothing)
function __init__()
    return VideoEditor.registerhdrdenoiser!(:oidn) do frame, image, effect
        d = DEFAULT[]
        if d === nothing
            d = OIDNDenoiser();DEFAULT[] = d
        end
        result = OIDNRunner.denoise(
            d, image; albedo = frame.albedo, normal = frame.normal,
            clean_aux = !effect.prefilter
        )
        [
            RGB{Float32}(result[x, y, 1], result[x, y, 2], result[x, y, 3])
                for x in axes(result, 1), y in axes(result, 2)
        ]
    end
end
function OIDNRunner.install!(d::OIDNDenoiser)
    VideoEditor.registerhdrdenoiser!(:oidn) do frame, image, effect
        result = OIDNRunner.denoise(
            d, image; albedo = frame.albedo, normal = frame.normal,
            clean_aux = !effect.prefilter
        )
        [
            RGB{Float32}(result[x, y, 1], result[x, y, 2], result[x, y, 3])
                for x in axes(result, 1), y in axes(result, 2)
        ]
    end
    return d
end
end
