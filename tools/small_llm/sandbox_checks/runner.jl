using Test, Sockets
@test !haskey(ENV,"SMALL_LLM_SANDBOX_SENTINEL")
@test !ispath("/dev/dri")
@test !ispath("/home/sim/.ssh")
@test !ispath("/home/sim/.codex")
@test !ispath("/home/sim/Programmieren/VulkanDev/dev/JuliaVision")
@test_throws SystemError open("/usr/sandbox-write-probe","w")
@test_throws SystemError open("/home/sim/Programmieren/VulkanDev/Project.toml","a")
@test_throws Exception connect(ip"1.1.1.1",80)
write("/work/writable-probe","ok")
@test read("/work/writable-probe",String)=="ok"
println("ISOLATION_CHECKS_PASSED")
