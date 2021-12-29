HOPPER_RANDOM_SCORE = -20.272305
HALFCHEETAH_RANDOM_SCORE = -280.178953
WALKER_RANDOM_SCORE = 1.629008
ANT_RANDOM_SCORE = -325.6

HOPPER_EXPERT_SCORE = 3234.3
HALFCHEETAH_EXPERT_SCORE = 12135.0
WALKER_EXPERT_SCORE = 4592.3
ANT_EXPERT_SCORE = 3879.7


def prepare_env(env_name):
    if env_name == 'door-human-v0':
        q_file_location = 'Model/door/door-human-v0/door-human-v0_q.pth'
        bc_file_location = 'Model/door/door-human-v0/door-human-v0_bc_standard.pth'
        q_pi_file_location = 'Model/door/door-human-v0/door-human-v0_q_pi.pth'
        actor_file_location = 'Model/door/door-human-v0/door-human-v0_actor.pth'
        env_name_ = 'door'

    if env_name == 'pen-human-v0':
        q_file_location = 'Model/pen/pen-human-v0/pen-human-v0_q.pth'
        bc_file_location = 'Model/pen/pen-human-v0/pen-human-v0_bc_standard.pth'
        q_pi_file_location = 'Model/pen/pen-human-v0/pen-human-v0_q_pi.pth'
        actor_file_location = 'Model/pen/pen-human-v0/pen-human-v0_actor.pth'
        env_name_ = 'pen'

    if env_name == 'hammer-human-v0':
        q_file_location = 'Model/hammer/hammer-human-v0/hammer-human-v0_q.pth'
        bc_file_location = 'Model/hammer/hammer-human-v0/hammer-human-v0_bc_standard.pth'
        q_pi_file_location = 'Model/hammer/hammer-human-v0/hammer-human-v0_q_pi.pth'
        actor_file_location = 'Model/hammer/hammer-human-v0/hammer-human-v0_actor.pth'
        env_name_ = 'hammer'

    # Hopper
    if env_name == 'hopper-random-v2':
        q_file_location = 'Model/hopper/hopper_random/hopper_random_v2_q.pth'
        bc_file_location = 'Model/hopper/hopper_random/hopper_random_v2_bc_standard.pth'
        q_pi_file_location = 'Model/hopper/hopper_random/hopper_random_v2_q_pi.pth'
        actor_file_location = 'Model/hopper/hopper_random/hopper_random_v2_actor.pth'
        env_name_ = 'hopper'

    if env_name == 'hopper-medium-v2':
        q_file_location = 'Model/hopper/hopper_medium/hopper_medium_v2_q.pth'
        bc_file_location = 'Model/hopper/hopper_medium/hopper_medium_v2_bc_standard.pth'
        q_pi_file_location = 'Model/hopper/hopper_medium/hopper_medium_v2_q_pi.pth'
        actor_file_location = 'Model/hopper/hopper_medium/hopper_medium_v2_actor.pth'
        env_name_ = 'hopper'

    if env_name == 'hopper-expert-v2':
        q_file_location = 'Model/hopper/hopper_expert/hopper_expert_v2_q.pth'
        bc_file_location = 'Model/hopper/hopper_expert/hopper_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/hopper/hopper_expert/hopper_expert_v2_q_pi.pth'
        actor_file_location = 'Model/hopper/hopper_expert/hopper_expert_v2_actor.pth'
        env_name_ = 'hopper'

    # if env_name == 'hopper-expert-v1':
    #     q_file_location = 'Model/hopper/hopper_expert_v1_q.pth'
    #     bc_file_location = 'Model/hopper/hopper_expert_v1_bc_standard.pth'
    #     q_pi_file_location = 'Model/hopper/hopper_expert_v1_q_pi.pth'
    #     actor_file_location = 'Model/hopper/hopper_expert_v1_actor.pth'
    #     env_name_ = 'hopper'
    #
    # if env_name == 'hopper-expert-v0':
    #     q_file_location = 'Model/hopper/hopper_expert_v0_q.pth'
    #     bc_file_location = 'Model/hopper/hopper_expert_v0_bc_standard.pth'
    #     q_pi_file_location = 'Model/hopper/hopper_expert_v0_q_pi.pth'
    #     actor_file_location = 'Model/hopper/hopper_expert_v0_actor.pth'
    #     env_name_ = 'hopper'

    if env_name == 'hopper-medium-expert-v2':
        q_file_location = 'Model/hopper/hopper_medium_expert/hopper_medium_expert_v2_q.pth'
        bc_file_location = 'Model/hopper/hopper_medium_expert/hopper_medium_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/hopper/hopper_medium_expert/hopper_medium_expert_v2_q_pi.pth'
        actor_file_location = 'Model/hopper/hopper_medium_expert/hopper_medium_expert_v2_actor.pth'
        env_name_ = 'hopper'

    if env_name == 'hopper-medium-replay-v2':
        q_file_location = 'Model/hopper/hopper_medium_replay/hopper_medium_replay_v2_q.pth'
        bc_file_location = 'Model/hopper/hopper_medium_replay/hopper_medium_replay_v2_bc_standard.pth'
        q_pi_file_location = 'Model/hopper/hopper_medium_replay/hopper_medium_replay_v2_q_pi.pth'
        actor_file_location = 'Model/hopper/hopper_medium_replay/hopper_medium_replay_v2_actor.pth'
        env_name_ = 'hopper'

    # Halfcheetah
    if env_name == 'halfcheetah-random-v2':
        q_file_location = 'Model/halfcheetah/halfcheetah_random/halfcheetah_random_v2_q.pth'
        bc_file_location = 'Model/halfcheetah/halfcheetah_random/halfcheetah_random_v2_bc_standard.pth'
        q_pi_file_location = 'Model/halfcheetah/halfcheetah_random/halfcheetah_random_v2_q_pi.pth'
        actor_file_location = 'Model/halfcheetah/halfcheetah_random/halfcheetah_random_v2_actor.pth'
        env_name_ = 'halfcheetah'

    if env_name == 'halfcheetah-medium-v2':
        q_file_location = 'Model/halfcheetah/halfcheetah_medium/halfcheetah_medium_v2_q.pth'
        bc_file_location = 'Model/halfcheetah/halfcheetah_medium/halfcheetah_medium_v2_bc_standard.pth'
        q_pi_file_location = 'Model/halfcheetah/halfcheetah_medium/halfcheetah_medium_v2_q_pi.pth'
        actor_file_location = 'Model/halfcheetah/halfcheetah_medium/halfcheetah_medium_v2_actor.pth'
        env_name_ = 'halfcheetah'

    if env_name == 'halfcheetah-expert-v2':
        q_file_location = 'Model/halfcheetah/halfcheetah_expert/halfcheetah_expert_v2_q.pth'
        bc_file_location = 'Model/halfcheetah/halfcheetah_expert/halfcheetah_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/halfcheetah/halfcheetah_expert/halfcheetah_expert_v2_q_pi.pth'
        actor_file_location = 'Model/halfcheetah/halfcheetah_expert/halfcheetah_expert_v2_actor.pth'
        env_name_ = 'halfcheetah'

    if env_name == 'halfcheetah-medium-expert-v2':
        q_file_location = 'Model/halfcheetah/halfcheetah_medium_expert/halfcheetah_medium_expert_v2_q.pth'
        bc_file_location = 'Model/halfcheetah/halfcheetah_medium_expert/halfcheetah_medium_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/halfcheetah/halfcheetah_medium_expert/halfcheetah_medium_expert_v2_q_pi.pth'
        actor_file_location = 'Model/halfcheetah/halfcheetah_medium_expert/halfcheetah_medium_expert_v2_actor.pth'
        env_name_ = 'halfcheetah'

    if env_name == 'halfcheetah-medium-replay-v2':
        q_file_location = 'Model/halfcheetah/halfcheetah_medium_replay/halfcheetah_medium_replay_v2_q.pth'
        bc_file_location = 'Model/halfcheetah/halfcheetah_medium_replay/halfcheetah_medium_replay_v2_bc_standard.pth'
        q_pi_file_location = 'Model/halfcheetah/halfcheetah_medium_replay/halfcheetah_medium_replay_v2_q_pi.pth'
        actor_file_location = 'Model/halfcheetah/halfcheetah_medium_replay/halfcheetah_medium_replay_v2_actor.pth'
        env_name_ = 'halfcheetah'

    # walker2d
    if env_name == 'walker2d-random-v2':
        q_file_location = 'Model/walker2d/walker2d_random/walker2d_random_v2_q.pth'
        bc_file_location = 'Model/walker2d/walker2d_random/walker2d_random_v2_bc_standard.pth'
        q_pi_file_location = 'Model/walker2d/walker2d_random/walker2d_random_v2_q_pi.pth'
        actor_file_location = 'Model/walker2d/walker2d_random/walker2d_random_v2_actor.pth'
        env_name_ = 'walker2d'

    if env_name == 'walker2d-medium-v2':
        q_file_location = 'Model/walker2d/walker2d_medium/walker2d_medium_v2_q.pth'
        bc_file_location = 'Model/walker2d/walker2d_medium/walker2d_medium_v2_bc_standard.pth'
        q_pi_file_location = 'Model/walker2d/walker2d_medium/walker2d_medium_v2_q_pi.pth'
        actor_file_location = 'Model/walker2d/walker2d_medium/walker2d_medium_v2_actor.pth'
        env_name_ = 'walker2d'

    if env_name == 'walker2d-expert-v2':
        q_file_location = 'Model/walker2d/walker2d_expert/walker2d_expert_v2_q.pth'
        bc_file_location = 'Model/walker2d/walker2d_expert/walker2d_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/walker2d/walker2d_expert/walker2d_expert_v2_q_pi.pth'
        actor_file_location = 'Model/walker2d/walker2d_expert/walker2d_expert_v2_actor.pth'
        env_name_ = 'walker2d'

    if env_name == 'walker2d-medium-expert-v2':
        q_file_location = 'Model/walker2d/walker2d_medium_expert/walker2d_medium_expert_v2_q.pth'
        bc_file_location = 'Model/walker2d/walker2d_medium_expert/walker2d_medium_expert_v2_bc_standard.pth'
        q_pi_file_location = 'Model/walker2d/walker2d_medium_expert/walker2d_medium_expert_v2_q_pi.pth'
        actor_file_location = 'Model/walker2d/walker2d_medium_expert/walker2d_medium_expert_v2_actor.pth'
        env_name_ = 'walker2d'

    if env_name == 'walker2d-medium-replay-v2':
        q_file_location = 'Model/walker2d/walker2d_medium_replay/walker2d_medium_replay_v2_q.pth'
        bc_file_location = 'Model/walker2d/walker2d_medium_replay/walker2d_medium_replay_v2_bc_standard.pth'
        q_pi_file_location = 'Model/walker2d/walker2d_medium_replay/walker2d_medium_replay_v2_q_pi.pth'
        actor_file_location = 'Model/walker2d/walker2d_medium_replay/walker2d_medium_replay_v2_actor.pth'
        env_name_ = 'walker2d'

    # elif env_name == 'halfcheetah-medium-v2':
    #     q_file_location = 'Model/hopper/halfcheetah_medium_v2_q.pth'
    #     bc_file_location = 'Model/hopper/halfcheetah_medium_v2_bc_standard.pth'
    #     q_pi_file_location = 'Model/hopper/halfcheetah_medium_v2_q_pi.pth'
    #     actor_file_location = 'Model/hopper/halfcheetah_medium_v2_actor.pth'

    return q_file_location, bc_file_location, q_pi_file_location, actor_file_location, env_name_
