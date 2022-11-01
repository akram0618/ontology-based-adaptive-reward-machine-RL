from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.reward_functions import *
import simpy
import numpy as np
from collections import defaultdict
from tensorforce.execution import Runner
from tensorforce.environments import Environment
from tensorforce import util, TensorforceError
from tensorforce.core.utils.json_encoder import NumpyJSONEncoder
#from production.envs.transport import transport
from collections import OrderedDict
from tensorforce import Agent, Environment
from tensorforce.environments import RemoteEnvironment


class Transport_order(Resource):
    all_transp_orders = []  # Overall list of available transport orders
    agents_waiting_for_action = []


    def __init__(self, env, id, resp_area, agent_type, statistics, parameters, resources, agents, time_calc, location, label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Transportation %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        #self.resp_area = resp_area
        self.type = "transp_ord"
        self.idle = env.event()
        self.transp_log = [["action", "sim_time", "from_at", "to_at", "duration"]]
        self.current_order = None
        # We add machines and orders
        self.machine_orders = defaultdict(list)
        self.transp_orders = defaultdict(list)
        self.agent_type = agent_type
        self.mapping = None
        self.counter = 0
        self.sum_reward = 0.0
        self.state_before = self.calculate_state()
        self.next_action = None
        self.latest_reward = 0.0
        self.next_action_order = None
        self.last_action_id = None
        self.last_reward_calc = 0.0
        self.last_reward_calc_time = 0.0
        self.agents_waiting_for_action.append(self)
        #print("agents_waiting_for_action: ",self.agents_waiting_for_action[0])

    def calculate_state(self):
        #print("self id : " , self.id)
        result_state = []
        state_type = 'bool'
        #waiting time
        state = [0.0] * 41
        self.transp_orders = [-1]* 40
        state[0] = len(Transport_order.all_transp_orders)
        if len(Transport_order.all_transp_orders) > 0:
            all_transp_orders_sorted = dict()
            for order in Transport_order.all_transp_orders:
                all_transp_orders_sorted[order] = order.get_total_waiting_time()
            all_transp_orders_sorted2 = sorted(all_transp_orders_sorted, key=all_transp_orders_sorted.get, reverse=True)
            all_transp_orders_sorted = all_transp_orders_sorted2[0:40]
            #print("all_transp_orders_sorted:",all_transp_orders_sorted)
        for i in range(min(len(Transport_order.all_transp_orders) , 40)):
            order = all_transp_orders_sorted[i]
            state[i+1] = order.get_total_waiting_time()
            self.transp_orders[i] = order
        result_state.extend(state)
        # due date
        state = [0.0] * 41
        state[0] = len(Transport_order.all_transp_orders)
        if len(Transport_order.all_transp_orders) > 0:
            all_transp_orders_sorted = dict()
            for order in Transport_order.all_transp_orders:
                all_transp_orders_sorted[order] = order.order_waiting_threshold - order.get_total_waiting_time()
            all_transp_orders_sorted2 = sorted(all_transp_orders_sorted, key=all_transp_orders_sorted.get, reverse=True)
            all_transp_orders_sorted = all_transp_orders_sorted2[0:40]
        for i in range(min(len(Transport_order.all_transp_orders), 40)):
            order = all_transp_orders_sorted[i]
            state[i + 1] = order.order_waiting_threshold - order.get_total_waiting_time()
        result_state.extend(state)
        # priority
        state = [0.0] * 41
        state[0] = len(Transport_order.all_transp_orders)
        for i in range(min(len(Transport_order.all_transp_orders), 40)):
            order = Transport_order.all_transp_orders[i]
            state[i + 1] = order.order_priority
        result_state.extend(state)
        return result_state

    def actions(self):
        return dict(type='int', shape=(1000))

    def calculate_reward(self, action):
        #print("transport.calculate_reward")
        result_reward = self.parameters['TRANSP_AGENT_REWARD_INVALID_ACTION']
        result_terminal = False
        result_reward1 = get_reward_order_waiting_time(self, action[0], result_reward)
        result_reward2 = get_reward_order_duedate(self, action[1])
        result_reward3 = get_reward_order_priority(self, action[2])
        self.latest_reward = result_reward
        return result_reward1,result_reward2,result_reward3