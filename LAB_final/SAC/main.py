from SAC_racecar import SACAgent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 256,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 3e-4,
		"lrc": 3e-4,
		"temperature_lr": 3e-4,
		"replay_buffer_capacity": 20000,
		"logdir": 'log/',
		"update_freq": 2,
		"eval_interval": 20,
		"eval_episode": 5,
		"init_temperature": 0.1,
		"game": "CarRacing-v2",    #"Pendulum-v1"  #racecar_gym #CarRacing-v2
		"observatrion_Image": True,
		"frames": 8,
		"already_time_step": 0
	}
	agent = SACAgent(config)
	#agent.load("./log/model_128169_149.pth")
	agent.train()
	#agent.record_video()
	#agent.evaluate()
	#agent.load_and_evaluate("./log/model_12333_0.pth")
	



