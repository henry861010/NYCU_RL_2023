# DQN to play Atari-MsPacman-v5
## Run Project  

## File
1. main.py: configurate the setting and execute the training, configurating:  
    1. `gpu`: (True/False), if provide the GPU to accelerate the training  
    2. `training_steps`: (int), how many the transactions play to train the model(not the episode!)  
    3. `gamma`: (float), the learning rate(used to calculate the target q-value)  
    4. `batch_size`: (int), the number of the transactions we want to select from replay buffer to update the DNN   
    5. `warmup_steps`: (int), the minimum step before update the DNN  
    6. `eps_min`: (int), minimum epsilon(epsilon wouldn't decline under this value)  
    7. `eps_decay`: (int), the decline rate of the epsilon  
    8. `eval_epsilon`: (float),  
    9. `replay_buffer_capacity`: (int), the capacity if the replay buffer  
    10. `logdir`: (string), log directory  
    11. `update_freq`: (int), how many the transactions play to update behavier network  
    12. `update_target_freq`: (int), how many the transactions play to update target network  
    13. `learning_rate`: (int), the learning rate of the DNN(how much the property from the loss will update to the network)
    14. `eval_interval`: (int), the frequency to reocrd the performance  
    15. `eval_episode`: (int), the range to reocrd the performance  
    16. `env_id`: (string), the enviroment model id. in this project the id is "ALE/MsPacman-v5"  
2. base_agent.py: the RL training model which include the function:  
    1. `update`: which call the function `update_behavior_network` and `update_target_network` to update the DNN when the specified time reach
    2. `epsilon_decay`: help the model to converge
    3. `train`: the main procedure to train the model
    4. `evaluate`: record the performance
    5. `save` and `load`: used to save/load the dnn weight
3. dqn_agent_atari.py: to override the function `decide_agent_actions` and `update_behavior_network` in base_agent.py
4. replay_buffer/replay_buffer.py: the experience buffer used to record all the history. if we update the DNN after each episode terminated immediately, there is negative effect arise because the relationship of the transactions in the same episode. to avoud that, we update the DNN by smaple the transactions from the replay_buffer separately.
5. models/atari_model.py: the DNN network used to approximate the q-value.  
## Note

## Reference
1. example of DQL by PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?fbclid=IwAR3VXhsV3taRYrsBhgM1Un7KtH8_PntO7dfiJiTegFEUG7CdR1BBirgJhRA  
2. example of DNN by PyTorch: https://medium.com/pyladies-taiwan/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E6%96%B0%E6%89%8B%E6%9D%91-pytorch%E5%85%A5%E9%96%80-511df3c1c025  
3. library - gym: a library provided by openAI which provide the enviroment of Atri game to develop the RL: https://www.gymlibrary.dev/index.html  
4. the basic usage of the pyTorch-tensor: https://www.cupoy.com/marathon-mission/00000175AB50EF1A000000016375706F795F72656C656173654355/00000175C0DF5194000000016375706F795F72656C656173654349/  
5. the RL concept: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95/
https://zhuanlan.zhihu.com/p/336723691  
6. tensor: https://www.cupoy.com/marathon-mission/00000175AB50EF1A000000016375706F795F72656C656173654355/00000175C0DF5194000000016375706F795F72656C656173654349/ 
