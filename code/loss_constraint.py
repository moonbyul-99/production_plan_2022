import numpy as np
import pandas as pd



def color_loss(up_color, body_color):
    l = len(body_color)
    count = 0
    state = up_color[0]
    t = 0
    
    for i in range(0,l):
        if state == up_color[i]: # 颜色一致，不切换
            t += 40
            if state == body_color[i]: # 颜色一致，不切换
                t+= 40
                count += 1
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
'''
# 涂装车间用时
def color_loss(up_color, body_color):
    # up_color为加工计划的车顶颜色     黑曜黑，石黑，无对比颜色，特殊颜色（绿色）事先将无对比颜色转为对应的车身颜色
    # body_color 为加工计划的车身颜色  薄雾灰，天空灰，飞行蓝，水晶紫，水晶珍珠白，明亮红，闪耀黑
    l = len(body_color)
    count = 0
    state = up_color[0]
    t = 0 

    #前5辆车不需要强制清洗喷头
    for i in range(0,5):
        if state == up_color[i]:
            t+=40
        else:
            state = up_color[i]
            t += 120
        if state == body_color[i]:
            t += 40
        else: 
            t += 120
            state = body_color[i]
            
    for i in range(5,l):
        #判断是否为5k辆车，决定是否需要强制清洗喷头
        if i%5 == 0:
            # 清洗喷头时间 加 车顶时间
            t+= 120
            state =up_color[i] #切换喷头状态
            if state == body_color[i]:
                t += 40
            else:
                state = body_color[i]
                t += 120

        else: #不需要强制清洗喷头
            if state == up_color[i]: #不需要切换喷头状态
                t += 40
            else: # 喷头状态与车顶颜色不一致，切换
                state = up_color[i]
                t += 120

            if state == body_color[i]:#车身车顶一致，不用切换
                t += 40
            else:
                state = body_color[i]
                t += 120
    return t
'''
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

#总用时
def loss(plan, data):
    #plan 为一个排列，代表生产顺序
    #data 为原始数据，记录生产信息
    car_type = list(data['车型'].iloc[plan])
    up_color = list(data['车顶颜色'].iloc[plan])
    body_color = list(data['车身颜色'].iloc[plan])

    t_1 = welding_loss(car_type)
    t_2 = color_loss(up_color,body_color)
    t_3 = 80*len(plan)
    t = t_1 + t_2 + t_3
    return (t_1,t_2,t_3,t)

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
