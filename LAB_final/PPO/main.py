from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e10,
		"update_sample_count": 2000,#10000
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.4,
		"batch_size": 1024,
		"logdir": 'log/Enduro/',
		"update_ppo_epoch": 10,  #3
		"learning_rate": 2e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.035,
		"horizon": 1024,
		"eval_interval": 70,
		"eval_episode": 5,  #####
		"frames_stack": 8,
		"total_time_step": 0,
	}
	# how to adjust hyperparameter: https://zhuanlan.zhihu.com/p/345353294 

	actions = [
		[-1, 1.0],[0, 1.0],[1, 1.0],
        [-1, 0.4],[0, 0.4],[1, 0.4],
		[-1, 0.0],[0, 0.0],[1, 0.0],
		[-1,-0.4],[0,-0.4],[1,-0.4],
        [-1,-1.0],[0,-1.0],[1,-1.0]
	]

	agent = AtariPPOAgent(config, actions)
	#agent.load(".\\log\Enduro\\austin_map\\") 
	agent.train()
	#agent.record_video()
	#agent.load_and_evaluate(".\\log\Enduro\\aus_model_4330201_550.pth")
 


