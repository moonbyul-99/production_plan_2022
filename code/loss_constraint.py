import numpy as np
import pandas as pd

def color_loss(up_color, body_color,r = 0):
#当出现连续生产时, 喷漆时间修正为40/(1+r)**count  r >= 0
    l = len(body_color)
    count = 0
    state = up_color[0]
    t = 0

    for i in range(0,l):
        if state == up_color[i]: # 颜色一致，不切换
            t += 40/(1+r)**count
            if state == body_color[i]: # 颜色一致，不切换
                count += 1
                t += 40/(1+r)**count
                if count == 5: #连续5辆，清洗
                    count = 0
                    t += 80
            else: # 颜色不一致，清洗切换
                t += 120
                count = 0
                state = body_color[i]
        else:  #颜色不一致，清洗切换
            state = up_color[i]
            t += 120
            count = 0
            if state == body_color[i]:
                t += 40
                count += 1
            else:
                t += 120
                state = body_color[i]
                count = 0
    return t


#焊装车间用时
def welding_loss(car_type):
    # car_type 为加工计划的车型列表[A,B...]
    # 加工顺序从左至右
    state = car_type[0]
    n = 0
    t = 0
    #res = []
    change = 0
    count = 0
    while n <  len(car_type):
        #判断是否需要切换
        if state == car_type[n]:
            # 生产状态与车型一致，不需要切换
            count += 1
            t += 80
        else:
            #生产状态与车型不一致，需要切换，同时需要判断能否立即切换
            if change == 1:
                #此时可以立即切换
                count = 1 # 一个周期内的次数重置
                t += 80
                state = car_type[n]
            else:
                #此时不能立即切换需要等待到30min后再切换
                state = car_type[n]
                #计算等待完成切换的时间
                t += 80 + 1800 - 80*count
                count = 1
        n += 1
    return t 

#总装车间用时
def install_loss(engine_type,r = 0):
#engine_type 记录四驱车，两驱车信息
    l = len(engine_type)
    count = 0
    t = 0
    if r == 0:#不考虑正则
        t = 80*l
    else:
        for i in range(0,l):
            if engine_type[i] == "两驱":
                count = 0
                t += 80
            else:
                count += 1
                t += 80*((1+r)**max(count-4,0))
    return t


#总用时
def loss(plan, data,r0 = 0,r1 = 0):
    #plan 为一个排列，代表生产顺序
    #data 为原始数据，记录生产信息
    car_type = list(data['车型'].iloc[plan])
    up_color = list(data['车顶颜色'].iloc[plan])
    body_color = list(data['车身颜色'].iloc[plan])
    engine_type = list(data['变速器'].iloc[plan])

    t_1 = welding_loss(car_type)
    t_2 = color_loss(up_color,body_color,r0)
    t_3 = install_loss(engine_type,r1)
    t = t_1 + t_2 + t_3
    return (t_1,t_2,t_3,t)


'''
#发动机约束 判断当前的约束是否满足发动机类型约束
def engine_cons(engine_type):
    l = len(engine_type)
    count = 0
    for i in range(0,l):
        if engine_type[i] == '四驱':
            count += 1
        if count >=4:
            res = False
            #print(i)
            break
        else:
            res = True
        if engine_type[i] == '两驱':
            count = 0
    return res
'''
