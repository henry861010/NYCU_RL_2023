using the RL - TD to play the game 2048
# require of the lab1
modify the code in 2048_sample.cpp, in line:
1. #465: `virtual float estimate(const board& b) const {}` : to calculate the value of the board under the feature with 8 isomorphic. 
2. #473: `virtual float update(const board& b, float u) {}` :
3. #511: `size_t indexof(const std::vector<int>& patt, const board& b) const {}` : feature is the smaller board in the 4x44 board 
4. #678: `state select_best_move(const board& b) const {}` :
5. #709: `void update_episode(std::vector<state>& path, float alpha = 0.1) const {}` :
# concept
1. feature is the smaller board in the 4x4 board. for example feature [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] represents the 4x4 one, [0,1,2,3] represents top row and [0,4,8,12] most left column. because of the limited size of memory to record all the estimated value in different borad condition, we using one or multiple features to represent entire condition(which is the approximative method).
2. pattern:feature, pattern content 8/4/1 features, they are varient from the origin board through map(x2) or/and rotation(x4) operations. and call them "isomorphic". fore example board: 0  1  2  3, 4  5  6  7, 8  9  10 11, 12 13 14 15 with feature: 0 1 2  will has 8 isomorphic (1) 0 1 2, 4 5 6  (2) 3 7 11, 2 6 10 (3) 15 14 13, 11 10 9 (4) 12 8 4, 13 9 5 (5) 3 2 1, 7 6 5 (6) 15 11 7, 14 10 6 (7) 12 13 14, 8  9  10  (8) 0 4 8, 1 5 9. the store in isomorphic[] in the pattern class.
3. how can we number the different board(feature) condition? using function `indexof()` which index each condition by arranging all the element'value of condition in a unit variable.for example, in 2*2 table, we can nmuber a feature with value [1,2,4,2] as 0x1242. in project `indexof(isomorphic[i], b)`
4. the value of the board(features) = sum of each features's ( sum ( VALUE_OF_ISORMORPHIC ) in features / NUMBER_ISOMORPHIC_OF_THE_FEATURE ) / NUMBER_OF_FEATURES
   `sum for each feature and each isomorphic ~ weight[indexof(isomorphic[i], b)];`
#
