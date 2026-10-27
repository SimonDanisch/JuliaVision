using Test, Trellis2Runner, Mantle
include("test_decodergraph.jl")
test_sparse_decoder_graph(Trellis2Runner,Mantle.Device())
