using Test
using TrackingLoops
# Internal, used by the bandwidth tests.
using TrackingLoops: MAX_CODE_LOOP_BANDWIDTH_TIME_PRODUCT
using GNSSSignals
using StaticArrays
using Unitful
using Unitful: Hz, dBHz, s, ms, @u_str

# The wide carrier bandwidth as capped for a record of `integration_time`.
wide_cap(bandwidth, integration_time) = TrackingLoops._capped_bandwidth(
    bandwidth,
    integration_time,
    TrackingLoops._MAX_WIDE_CARRIER_LOOP_BANDWIDTH_TIME_PRODUCT,
)
using AllocCheck
using LinearAlgebra: norm
using TrackingLoopFilters: ThirdOrderAssistedBilinearLF, SecondOrderBilinearLF

include("correlators.jl")
include("sample_parameters.jl")
include("noise_estimators.jl")
include("nco_timeline.jl")
include("estimators.jl")
include("vector_estimator.jl")
include("carrier_loop_staging.jl")
include("signal_combining.jl")
include("vector_model.jl")
include("vector_simulation.jl")
include("vector_tracking.jl")
include("vector_engine.jl")
include("vector_closed_loop.jl")
include("vector_pipeline.jl")
include("vector_signal_groups.jl")
include("record.jl")
include("signal_state.jl")
include("bit_buffer.jl")
include("signals.jl")
include("cn0_estimators.jl")
include("discriminators.jl")
include("post_corr_filter.jl")
include("loop_filters.jl")
include("allocations.jl")
include("vector_allocations.jl")
