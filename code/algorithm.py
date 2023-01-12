# 算法
import numpy as np
import pandas as pd
import loss_constraint as lc

#-----------------------------------------
#遗传算法

#生成满足约束的一组解

def generator(data,N):
    n = data.shape[0]
#  使用numpy 操作代替list 提高速度
    population = np.apply_along_axis(lambda x: np.random.permutation(n),axis=1,arr =np.zeros((N,n)))
    return population


#N 为群体数目  T为迭代次数  alpha  为群体中直接保留的样本比例  beta 为群体中生成后代的样本比例 alpha + beta <= 1
#seed_parent 一组优化的较好地父本
def GA(data,N,T,alpha,beta,use_seed = False, seed_parent = None):
 
    if use_seed :
        population = np.zeros((N,data.shape[0]),dtype = int)
        n_0 = seed_parent.shape[0]
        population[0:n_0,:] = seed_parent
        population[n_0:N,:] = generator(data,N-n_0)
    else:
        population = generator(data,N)

    n_1 = int(alpha*N)
    n_2 = int(beta*N)
    n_3 = N - n_1 - n_2
    mean_cost = data.shape[0]*3000
    best_cost = data.shape[0]*3000
    mean_record = []
    best_record = []
    for t in range(0,T):
        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(N)
        x = np.min(cost)
        if x < best_cost:
            best_cost = x

        mean_cost = np.mean(cost)
        best_record.append(best_cost)
        mean_record.append(mean_cost)
        # 保留前n1个最佳样本的id
        sel_id = np.argpartition(np.array(cost), n_1)[0:n_1]
        best_population = population[sel_id,:]

        # 群体中生成n2个新的样本
        children = get_child(population, data, n_2)
        # 群体中引入n3个全新的样本
        new_population = generator(data, n_3)

        if t%50 == 0:
            print(str(t) + '  iteration' + '     best loss' + str(best_cost))

        #population = best_population + children + new_population
        population[0:n_1,:] = best_population
        population[n_1:n_1 +n_2,:] = children
        population[n_1+n_2:N,:] = new_population
    
    return population, mean_cost, best_cost, mean_record, best_record

'''
def GA(data,N,T,alpha,beta):
    population = generator(data,N)

    n_1 = int(alpha*N)
    n_2 = int(beta*N)
    n_3 = N - n_1 - n_2
    mean_cost = data.shape[0]*3000
    best_cost = data.shape[0]*3000
    mean_record = []
    best_record = []
    for t in range(0,T):
        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(N)
        x = np.min(cost)
        if x < best_cost:
            best_cost = x

        mean_cost = np.mean(cost)
        best_record.append(best_cost)
        mean_record.append(mean_cost)
        # 保留前n1个最佳样本的id
        sel_id = np.argpartition(np.array(cost), n_1)[0:n_1]
        best_population = population[sel_id,:]

        # 群体中生成n2个新的样本
        children = get_child(population, data, n_2)
        # 群体中引入n3个全新的样本
        new_population = generator(data, n_3)

        if t%50 == 0:
            print(str(t) + '  iteration' + '     best loss' + str(best_cost))

        #population = best_population + children + new_population
        population[0:n_1,:] = best_population
        population[n_1:n_1 +n_2,:] = children
        population[n_1+n_2:N,:] = new_population
    
    return population, mean_cost, best_cost, mean_record, best_record
'''

#从群体中生成一组新的样本
def get_child(population, data, n_2):
    children = []
    N = population.shape[0]
    n = data.shape[0]   #序列长度
    parent_id = np.random.choice(N, n_2)
    parent_set = population[parent_id,:]
    children = np.apply_along_axis(lambda x: exchange(x),axis=1,arr = parent_set)
    '''
    x = np.array([i for i in range(0,n-1)])
    while count < n_2:

        parent = population[parent_id[count]]
        cut = np.sort(np.random.choice(x,2,replace= False))
        u = cut[0]
        v = cut[1]
        child = parent[0:u] +  parent[v:] + parent[u:v] 
        children.append(child)
        count += 1
    '''
    return children

#将一条序列截断为三段，123-> 231
def exchange(parent):
    n = parent.shape[0]
    x = np.array([i for i in range(0,n-1)])
    cut = np.sort(np.random.choice(x,2,replace= False))
    child = np.zeros(n,dtype = int)
    child[0:cut[1] - cut[0]] = parent[cut[0]:cut[1]] 
    child[cut[1] - cut[0]:n-cut[0]]  =  parent [cut[1]:] 
    child[n-cut[0]:] =  parent[0:cut[0]]
    return child



'''
def generator(data,N):
    n = data.shape[0]
    population = []
    count = 0

'''
    #if cons == 1:
    #    while count < N:
    #        plan = list(np.random.permutation(n))
    #        engine_type = list(data['变速器'].iloc[plan])
    #        if lc.engine_cons(engine_type):
    #            population.append(plan)
    #            count += 1
'''

#  直接约束四驱车比较困难，样本生成较慢，尝试不做约束
    while count < N:
        plan = list(np.random.permutation(n))
        population.append(plan)
        count += 1
    return population

'''

'''
#N 为群体数目  T为迭代次数  alpha  为群体中直接保留的样本比例  beta 为群体中生成后代的样本比例 alpha + beta <= 1
def GA(data,N,T,alpha,beta):
    population = generator(data,N)
    n_1 = int(alpha*N)
    n_2 = int(beta*N)
    n_3 = N - n_1 - n_2
    mean_cost = data.shape[0]*3000
    best_cost = data.shape[0]*3000
    mean_record = []
    best_record = []
    for t in range(0,T):

        cost = np.apply_along_axis(lambda x: lc.loss(x,data),axis=1,arr = population)[:,3].reshape(50000)
'''
        #cost = []
        #for i in range(0,N):
        #    plan = population[i]
        #    cost.append(lc.loss(plan, data)[3])
        #x = min(cost)
'''
        x = np.min(cost)
        if x < best_cost:
            best_cost = x
        #mean_cost = sum(cost)/N
        mean_cost = np.mean(cost)
        best_record.append(best_cost)
        mean_record.append(mean_cost)
        # 保留前n1个最佳样本的id
        sel_id = np.argpartition(np.array(cost), n_1)[0:n_1]
        #best_population = population[sel_id]
        best_population = [population[id] for id in sel_id]

        # 群体中生成n2个新的样本
        children = get_child(population, data, n_2)
        # 群体中引入n3个全新的样本
        new_population = generator(data, n_3)

        if t%50 == 0:
            print(str(t) + '  iteration' + '     best loss' + str(best_cost))

        population = best_population + children + new_population
    
    return population, mean_cost, best_cost, mean_record, best_record
'''

'''
#从群体中生成一组新的样本
def get_child(population, data, n_2):
    children = []
    N = len(population)
    n = data.shape[0]
    count = 0 
    parent_id = np.random.choice(N, n_2)
    x = np.array([i for i in range(0,n-1)])
    while count < n_2:

        parent = population[parent_id[count]]
        cut = np.sort(np.random.choice(x,2,replace= False))
        u = cut[0]
        v = cut[1]
        child = parent[0:u] +  parent[v:] + parent[u:v] 
'''
        #删除四驱约束
        #engine_type = list(data['变速器'].iloc[child])
        #while lc.engine_cons(engine_type) == False:
        #    cut = np.sort(np.random.choice(x,2,replace= False))
        #    u = cut[0]
        #    v = cut[1]
        #    child = parent[0:u] +  parent[v:] + parent[u:v] 
         #   engine_type = list(data['变速器'].iloc[child]
'''
        children.append(child)
        count += 1
    return children
'''           


