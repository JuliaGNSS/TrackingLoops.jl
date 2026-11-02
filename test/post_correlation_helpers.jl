# A post-correlation state whose latest integration is given rather than processed, as a
# host without `process_correlator_bank_output!` sets it: unsynchronised, no soft bits, no
# C/N₀ estimate unless one is passed. The keyword copy constructor does the work.
processed_state(
    signal,
    filtered_correlator_bank,
    previous_prompt,
    output::CorrelatorBankOutput,
    num_code_periods,
    sampling_frequency;
    command_issue_sample_index = output.sample_index,
    prn = 0,
    sample_index_offset = 0,
    carrier_phase_polarity = 0,
    cn0_estimator = NoCN0Estimator(),
) = processed_state(
    signal,
    filtered_correlator_bank,
    previous_prompt,
    output.num_integrated_samples,
    output.sample_index + sample_index_offset,
    command_issue_sample_index + sample_index_offset,
    num_code_periods,
    sampling_frequency;
    prn,
    code_phase = output.code_phase,
    carrier_phase_polarity,
    cn0_estimator,
)

processed_state(
    signal,
    filtered_correlator_bank,
    previous_prompt,
    num_integrated_samples::Integer,
    sample_index::Integer,
    command_issue_sample_index::Integer,
    num_code_periods,
    sampling_frequency;
    prn = 0,
    code_phase = NaN,
    carrier_phase_polarity = 0,
    cn0_estimator = NoCN0Estimator(),
) = PostCorrelationState(
    PostCorrelationState(signal, prn; correlator_bank = filtered_correlator_bank, cn0_estimator);
    sampling_frequency,
    last_filtered_correlator_bank = filtered_correlator_bank,
    previous_filtered_prompt = previous_prompt,
    last_num_integrated_samples = num_integrated_samples,
    last_sample_index = sample_index,
    last_command_issue_sample_index = command_issue_sample_index,
    last_num_code_periods = num_code_periods,
    last_code_phase = code_phase,
    last_carrier_phase_polarity = carrier_phase_polarity,
)
