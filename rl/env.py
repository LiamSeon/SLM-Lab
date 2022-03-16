from turtle import st
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
import os


class BaiscStockEnv(gym.Env):
    def __init__(self, dir:str, window_size:int, mode:str = 'csv', do_fast:bool = True):
        self.files_name = os.listdir(dir)
        self.dir_path = dir
        self.window_size = window_size
        self.mode = mode
        self.do_fast = do_fast
        if self.do_fast:
            self.load_files()
        
    def load_files(self):
        self.file_list = []
        print("\n\n\n########Loading Files########\n")
        for f in tqdm(self.files_name):
            if self.mode == 'csv':
                file = pd.read_csv(os.path.join(self.dir_path, f))
            elif self.mode == 'excel':
                file = pd.read_excel(os.path.join(self.dir_path, f)) # open dataframe files     
            self.file_list.append(file)

    def reset(self):
        super().reset()
        if not self.do_fast:
            self.file_name = np.random.choice(self.files_name, 1, replace = False)[0] # 무작위로 파일 오픈 순서 배정
        else:
            self.file_index = np.random.randint(0, len(self.file_list))

        self.current_time_index = 0 # 현재 열린 dataframe에서 self.start_index + self.current_time_index 위치를 나타냄. self.window_size 보다 클 수 없음.
        self.model_money = 10000 # model로 매도 매수 했을 시 남은 예산
        self.human_money = 10000 # holding 했을시의 남은 예산
        self.model_history = [self.model_money]
        self.human_history = [self.human_money]
        self.file_open()

    def file_open(self):
        if self.do_fast:
            self.current_file = self.file_list[self.file_index]
        elif self.mode == 'csv':
            self.current_file = pd.read_csv(os.path.join(self.dir_path, self.file_name)) # open dataframe files
        elif self.mode =='excel':
            self.current_file = pd.read_excel(os.path.join(self.dir_path, self.file_name)) # open dataframe files     
        self.start_index = random.randint(0, self.current_file.shape[0] - self.window_size) # randomly choose start index

    def next(self): #return done
        if self.current_time_index == self.start_index + self.window_size - 1:
            return True
        else:
            self.current_time_index += 1
            return False


    def step(self, action:int):
        # step => state, reward, done
        # action이 policy net
        done = self.next()
        state = self.current_file.iloc[self.current_time_index].to_numpy()
        y = state[-1]
        reward = 1
        
        return state, reward, done 

    def render(self):
        plt.plot(self.model_history)
        plt.plot(self.human_history)
        plt.show()


if __name__ == '__main__':
    print("starting environment test")
    DIR_PATH = "/Users/seonseung-yeob/Downloads/20220308222144"
    stock_env = BaiscStockEnv(dir = DIR_PATH, window_size = 10, mode = 'excel')
    for i in range(1000):
        stock_env.reset()