import random
import math
from typing import Callable
from time import monotonic
import logging

import matplotlib.pyplot as plt


DIMENSION = 5


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s'
)


def task_sphere(x: list[float]): # x - inf -> 0
    result = 0
    for i in range(DIMENSION):
        result += x[i]**2
    return result


def task_rozenbok(x: list[float]): # x - inf -> 0
    result = 0
    for i in range(DIMENSION - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (x[i]-1)**2
    return result


def task_stibinskiy_tang(x: list[float]): # x C [-5; 5] -> -39.16617*DIMENSION
    result = 0
    for i in range(DIMENSION):
        result += x[i]**4 - 16*x[i]**2 + 5*x[i]
    return 0.5*result


def logger(message: str, iter: int):
    if iter < 10 or iter < 100 and (iter + 1) % 10 == 0 or iter < 1000 and (iter + 1) % 100 == 0 or iter < 10000 and (iter + 1) % 1000 == 0 or iter % 10000 == 0:
        logging.info('iteration: {}\t\t{}'.format(iter + 1, message))


class SSA(object):
    def __init__(
        self, func,
        is_have_answer: bool = False,
        true_value: float = 0,
        epsilon: float = 0.2,
        left_bound: float= -1000,
        right_bound: float = 1000,
        max_iter: int = 5000,
        population_size: int = 200,
        
    ):
        self.__population_size: int = population_size
        self.__max_iter: int = max_iter
        self.__func = func
        self.__left_bound = left_bound
        self.__right_bound = right_bound

        self.is_have_answer = is_have_answer
        self.true_value = true_value
        self.epsilon = epsilon

        # инициализация популяции
        self.__population: list[list[float]] = list()

        for _ in range(self.__population_size):
            spider: list[float] = list()
            for _ in range(DIMENSION):
                spider.append(random.uniform(self.__left_bound, self.__right_bound))
            self.__population.append(spider)
        
        # инициализация типов пауков
        female_cnt = math.floor((0.9 - random.uniform(0, 0.25))*self.__population_size)
        dominant_cnt = math.floor((0.7 - random.uniform(0, 0.25))*(self.__population_size - female_cnt))
        nondominant_cnt = self.__population_size - female_cnt - dominant_cnt
        
        self.__spider_types: list[str] = list()
        for _ in range(female_cnt):
            self.__spider_types.append('F')
        for _ in range(dominant_cnt):
            self.__spider_types.append('D')
        for _ in range(nondominant_cnt):
            self.__spider_types.append('ND')
         
        # инициализация радиуса для спаривания
        self.__radius: float = (self.__right_bound - self.__left_bound)/(self.__population_size) # подумать


        # инициализация приспособленностей и крайних пауков
        self.__fitable_list: list[float] = list()
        
        self.__the_best_spider: list[float] = list() # вытащить по всем итерациям
        self.__the_worst_spider: list[float] = list()
        
        for spider in self.__population:
            current_spider_fit = self.__get_fit(spider)
            self.__fitable_list.append(current_spider_fit)
            if current_spider_fit >= max(self.__fitable_list):
                self.__the_best_spider = spider
            if current_spider_fit <= min(self.__fitable_list):
                self.__the_worst_spider = spider
        
        # инициализация крайних приспособленностей
        self.__best_fit: float = self.__get_best_fit()
        self.__worst_fit: float = self.__get_worst_fit()
        
        # инициализация крайних приспособленностей по всем итерациям
        self.__best_fit_list: list[float] = list()
        self.__worst_fit_list: list[float] = list()
        
        
       
    def run(self):
        start_time = monotonic()
        best_fit, worst_fit = self.__update_best_and_worst_fit_lists()
        logging.info('INITIALIZATION: best fit = {}, worst fit = {}'.format(best_fit, worst_fit))

        for iter in range(self.__max_iter):
            self.__move_female()
            self.__move_non_dominant()
            self.__move_dominant()

            best_fit, worst_fit = self.__update_best_and_worst_fit_lists()
            logger(f'best fit = {best_fit}, worst fit = {worst_fit}', iter)
            
            # для задачи на минимум
            if self.is_have_answer:
                if math.fabs(-1*best_fit - self.true_value) < self.epsilon:
                    break
            
        logging.info('MIN = {}, spider = {}'.format(-1*best_fit, self.__the_best_spider))
        logging.info('WORK TIME = {}'.format(monotonic() - start_time))
    
        plt.plot([i for i in range(1, len(self.__best_fit_list) + 1)], self.__best_fit_list, 'g', label='maximum')
        plt.show()
        
        

    def __move_female(self):
        for i in range(self.__population_size):
            if self.__spider_types[i] == 'F':

                next_point_female = self.__get_next_point_female(self.__population[i])
                self.__population[i] = next_point_female
                self.__fitable_list[i] = self.__get_fit(self.__population[i]) # считаем его новую приспособленность
                
                self.__update_the_best_and_worst_spiders(i)
    
    def __move_non_dominant(self):
        for i in range(self.__population_size):
            if self.__spider_types[i] == 'ND':
            
                next_point_male = self.__get_next_point_non_dominant(self.__population[i])
                
                self.__population[i] = next_point_male
                self.__fitable_list[i] = self.__get_fit(self.__population[i]) # считаем его новую приспособленность
                self.__update_the_best_and_worst_spiders(i)
    
    def __move_dominant(self):
        for i in range(self.__population_size):
            if self.__spider_types[i] == 'D':
                next_point_male = self.__get_next_point_dominant(self.__population[i])
                
                if self.__get_fit(next_point_male) > self.__get_fit(self.__population[i]):
                    self.__population[i] = next_point_male
                
                self.__fitable_list[i] = self.__get_fit(self.__population[i]) # считаем его новую приспособленность -> обновляем лучшие значения
                self.__update_the_best_and_worst_spiders(i)
    
    def __update_best_and_worst_fit_lists(self) -> tuple[float]:
        best_fit = self.__get_best_fit()
        worst_fit = self.__get_worst_fit()
        
        self.__best_fit_list.append(-1*best_fit)  
        self.__worst_fit_list.append(-1*worst_fit)
        
        return best_fit, worst_fit
    
    
    def __update_the_best_and_worst_spiders(self, i: int):
        if self.__fitable_list[i] >= self.__get_fit(self.__the_best_spider):
            self.__the_best_spider = self.__population[i]
        
        if self.__fitable_list[i] <= self.__get_fit(self.__the_worst_spider):
            self.__the_worst_spider = self.__population[i]

    def __get_fit(self, spider: list[float]) -> float:
        fit = -1*self.__func(spider)
        if self.__fitable_list != []:
            self.__get_best_fit()
            self.__get_worst_fit()
        return fit

    def __get_best_fit(self) -> float:
        self.__best_fit: float = max(self.__fitable_list)
        return self.__best_fit

    def __get_worst_fit(self) -> float:
        self.__worst_fit: float = min(self.__fitable_list)
        return self.__worst_fit

    def __get_weight(self, spider: list[float]) -> float:
        try:
            weight =  (self.__get_fit(spider) - self.__worst_fit)/(self.__best_fit - self.__worst_fit)
        except ZeroDivisionError:
            weight = 0
        return  weight

    def __get_population_without_the_spider(self, spider: list[float]):
        part_of_population = self.__population.copy()
        part_of_population.remove(spider)
        return part_of_population

    def __get_vibration(self, spider1: list[float], spider2: list[float]) -> float:
        vibration = self.__get_weight(spider2)*math.exp(
            -1*self.__get_dist_between_spiders(spider1, spider2)**2
        )
        return vibration

    def __get_dist_between_spiders(self, spider1: list[float], spider2: list[float]) -> float:
        sum_squar_delta = 0
        for i in range(DIMENSION):
            sum_squar_delta += (spider1[i]-spider2[i])**2
        
        return sum_squar_delta**0.5

    def __find_nearest_heavier(self, main_spider: list[float]) -> list[float]:

        part_of_population = self.__get_population_without_the_spider(main_spider)
        main_spider_weight = self.__get_weight(main_spider)
        
        min_dist = -1
        heavier_spider = main_spider
        for spider in part_of_population:
            if min_dist == -1 and self.__get_weight(spider) >= main_spider_weight:
                heavier_spider = spider
                min_dist = self.__get_dist_between_spiders(main_spider, spider)
            elif self.__get_weight(spider) >= main_spider_weight and min_dist >= self.__get_dist_between_spiders(main_spider, spider):
                heavier_spider = spider
                min_dist = self.__get_dist_between_spiders(main_spider, spider)
        
        return heavier_spider

    def __find_best_spider(self, spider: list[float] = None) -> list[float]:
        best_spider = self.__population[0]
        self.__best_fit = self.__get_fit(best_spider)
        for spider in self.__population:
            if self.__get_fit(spider) > self.__get_fit(best_spider):
                best_spider = spider
                self.__best_fit = self.__get_fit(best_spider)
        return best_spider

    def __find_nearest_female(self, spider: list[float]) -> list[float]:
        for i in range(self.__population_size):
            if self.__spider_types[i] == 'F' and self.__get_dist_between_spiders(spider, self.__population[i]) <= self.__radius:
                return self.__population[i]
        return [0]*DIMENSION

    def __get_next_point_female(self, spider: list[float]) -> list[float]:
        
        alpha = random.uniform(0, 1)
        beta = random.uniform(0, 1)
        delta = random.uniform(0, 1)

        nearest_heavier_spider = self.__find_nearest_heavier(spider)
        best_spider = self.__find_best_spider()
        
        vibration_from_nearest_heavier = self.__get_vibration_from_smb(self.__find_nearest_heavier, spider)
        vibration_from_best = self.__get_vibration_from_smb(self.__find_best_spider, spider)
     
        next_point = spider.copy()

        first_coeff = alpha*vibration_from_nearest_heavier
        second_coeff = beta*vibration_from_best

        delta_cur_near = list()
        delta_cur_best = list()
        for i in range(DIMENSION):
            next_point[i] += delta*(random.uniform(0, 1) - 0.5)
            delta_cur_near.append(first_coeff*(nearest_heavier_spider[i] - spider[i]))
            delta_cur_best.append(second_coeff*(best_spider[i] - spider[i]))

        probability = random.uniform(0, 1)

        if probability >= 0.5:
            for i in range(DIMENSION):
                next_point[i] += delta_cur_near[i] + delta_cur_best[i]
                next_point[i] = self.__check_bound(next_point[i])

        else:            
            for i in range(DIMENSION):
                next_point[i] -= delta_cur_near[i] + delta_cur_best[i]
                next_point[i] = self.__check_bound(next_point[i])
                
        return next_point

    def __get_vibration_from_smb(self, smb_finder: Callable, spider: list[float] = None) -> float:
        smb = smb_finder(spider)
        if smb == []:
            return 0
        vibration = self.__get_vibration(spider, smb)
        return vibration
    
    def __check_bound(self, next_point_i: float) -> float:
        if next_point_i > self.__right_bound:
            next_point_i = self.__right_bound
        if next_point_i < self.__left_bound:
            next_point_i = self.__left_bound
        return next_point_i
    
    def __get_next_point_dominant(self, spider: list[float]) -> list[float]:
        alpha = random.uniform(0, 1)
        delta = random.uniform(0, 1)

        next_point = spider.copy()
        female = self.__find_nearest_female(spider)
        vibration_from_female = self.__get_vibration_from_smb(self.__find_nearest_female, spider)
        
        for i in range(DIMENSION):
            next_point[i] += alpha*vibration_from_female*(female[i] - spider[i]) + delta*(random.uniform(0, 1) - 0.5)
            next_point[i] = self.__check_bound(next_point[i])
        
        return next_point

    def __get_next_point_non_dominant(self, spider: list[float]) -> list[float]:
        alpha = random.uniform(0, 1)
        next_point = spider.copy()
        all_weighted_spider_sum = [0 for _ in range(DIMENSION)]
        all_nd_spiders_weight = 0

        for i in range(self.__population_size):
            if self.__spider_types[i] == 'ND':
                nd_spider = self.__population[i]
                nd_spider_weight = self.__get_weight(nd_spider)
                all_nd_spiders_weight += nd_spider_weight

                for j in range(DIMENSION):
                    all_weighted_spider_sum[j] += nd_spider[j]*nd_spider_weight
        
        shift = list()
        
        for i in range(DIMENSION):
            try:
                all_weighted_spider_sum[i] /= all_nd_spiders_weight
            except ZeroDivisionError:
                all_weighted_spider_sum[i] = 0
            all_weighted_spider_sum[i] -= spider[i]
            shift.append(alpha*all_weighted_spider_sum[i])
            
        for i in range(DIMENSION):
            next_point[i] += shift[i]
            next_point[i] = self.__check_bound(next_point[i])

        return next_point
        

            # https://www.ijitee.org/wp-content/uploads/papers/v8i10/I8261078919.pdf
# https://www.hindawi.com/journals/mpe/2018/6843923/
    

if __name__ == '__main__':
    task_num = int(input(
        'Choose task:\n '
        '1 - task_sphere\n '
        '2 - task_rozenbok\n '
        '3 - task_stibinskiy_tang\n '
    ))
    if task_num == 1:
        SSA(
            func=task_sphere,
            left_bound=-10,
            right_bound=10,
            is_have_answer=True,
            true_value=0,
            epsilon=0.1,
            max_iter=5000,
            population_size=100,
        ).run()
    elif task_num == 2:
        SSA(
            func=task_rozenbok,
            left_bound=-3,
            right_bound=3,
            is_have_answer=True,
            true_value=0,
            epsilon=0.2,
            max_iter=5000,
            population_size=100,
        ).run()
    elif task_num == 3:
        SSA(
            func=task_stibinskiy_tang,
            left_bound=-5,
            right_bound=5,
            is_have_answer=True,
            true_value=-39.16617*DIMENSION,
            epsilon=0.1,
            max_iter=5000,
            population_size=100,
        ).run()
