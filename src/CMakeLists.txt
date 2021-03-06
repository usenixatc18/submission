include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${Boost_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS})

add_executable(schad 
    main.cpp
    packet/packet_builder.cpp
    configs/learning_config.cpp
    configs/policy_config.cpp
    configs/source_config.cpp
    configs/experiment_config.cpp
    configs/reward_config.cpp
    infrastructure/infrastructure_impl.cpp
    infrastructure/infrastructure.cpp
    traffic/poisson_source.cpp
    traffic/union_source.cpp
    traffic/sequence_source.cpp
    traffic/markov_source.cpp
    traffic/null_source.cpp
    policy/pq_policy.cpp
    reward/weighted_throughput_reward.cpp
    learning/epsilon_greedy.cpp
    learning/constant.cpp
    learning/ucb.cpp
    learning/ucb_e.cpp
    learning/combined.cpp
    learning/restarting.cpp
    learning/dgp_ucb.cpp
    learning/softmax.cpp
    learning/normalized.cpp
    learning/scaled.cpp
    learning/ucb_tuned.cpp
    simulator/simulation.cpp
    simulator/multi_run_stats_collector.cpp
    )

target_link_libraries(schad
    ${Boost_LIBRARIES})
