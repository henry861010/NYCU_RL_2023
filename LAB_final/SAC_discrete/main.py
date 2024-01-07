from SAC_racecar import SACAgent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 512,
		"warmup_steps": 600,
		"total_episode": 100000,
		"lra": 1e-3,
		"lrc": 1e-2,
		"temperature_lr": 1e-2,
		"replay_buffer_capacity": 20000,
		"logdir": 'log/',
		"update_freq": 2,
		"eval_interval": 70,
		"eval_episode": 5,
		"init_temperature": 0.1,
		"observatrion_Image": True,
		"frames": 8,
		"already_time_step": 0
	}
	
	actions = [
		[0.8,1],[1,0],[0.8,-1],[-0.6,0],[0,1],[0,-1]
	]

	agent = SACAgent(config,actions)
	#agent.load("./log/model_75899_99.pth")
	agent.train()
	#agent.record_video()
	#agent.evaluate()
	#agent.load_and_evaluate("./log/model_12333_0.pth")
	



