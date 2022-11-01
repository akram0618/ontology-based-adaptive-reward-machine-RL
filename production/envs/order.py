from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.transport import *
import simpy
import random

class Order(Resource):
    """An order specifices a production request.
    An order has a *id* and a sequence of *prod_steps* to fulfill.
    """

    def __init__(self, env, id, prod_steps, variant, statistics, parameters, resources, agents, time_calc):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, None)
        self.env = env
        self.id = id
        self.prod_steps = prod_steps
        self.variant = variant
        self.sop = -1
        self.eop = -1
        self.time_processing = 0
        self.time_handling = 0
        self.actual_step = 0
        self.finished = False
        
        # We add tardiness variable to order
        self.tardiness = False
        self.order_waiting_threshold = random.choices([5000, 3000, 1000], weights=(5, 70, 25), k=1)[0]
        ##############
        # We add order priority
        self.order_priority = random.choices([3, 2, 1], weights=(30, 50, 20), k=1)[0]

        ##############
        self.current_location = None
        self.order_log = [["action", "order_ID", "sim_time", "resource_ID"]]
        self.transported = self.env.event()
        self.processed = self.env.event()
        self.reserved = False
        if self.parameters['PRINT_CONSOLE']: print("Order %s created %s" % (self.id, [x.id for x in self.prod_steps]))
        self.order_log.append(["prod_steps", self.id, [x.id for x in prod_steps]])  # records the actual processing history incl time stamps of an order

    def set_sop(self):  # SOP = start of production
        self.sop = self.env.now
        self.statistics['stat_order_sop'][self.id] = self.sop
        self.statistics['stat_inv_episode'][-1][0] = self.env.now - self.statistics['stat_inv_episode'][-1][0]
        self.statistics['stat_inv_episode'].append([self.env.now, self.statistics['stat_inv_episode'][-1][1] + 1])
        self.order_log.append(["sop", self.id, round(self.sop, 5), ""])

    def set_eop(self):  # EOP = end of production
        self.eop = self.env.now
        self.statistics['stat_order_eop'][self.id] = self.eop
        self.statistics['stat_order_leadtime'][self.id] = self.eop - self.sop
        self.statistics['stat_inv_episode'][-1][0] = self.env.now - self.statistics['stat_inv_episode'][-1][0]
        self.statistics['stat_inv_episode'].append([self.env.now, self.statistics['stat_inv_episode'][-1][1] - 1])
        self.order_log.append(["eop", self.id, round(self.eop, 5), ""])

    def set_next_step(self):
        self.actual_step += 1
        if self.actual_step > len(self.prod_steps):
            self.finished = True

    def get_next_step(self):
        return self.prod_steps[self.actual_step]

    def get_total_waiting_time(self):
        result = self.env.now - self.sop - self.time_processing - self.time_handling
        return result

    def order_processing(self):
        while True:
            self.set_next_step()
            if self.finished:
                break

            if self.id >= 0 or self in self.current_location.buffer_out:  # Check initial orders that are created at the beginning
                self.order_log.append(["before_transport", self.id, round(self.env.now, 5), self.current_location.id])
                Transport.put(order=self, trans_agents=self.resources['transps'])
                yield self.transported  # Transport is finished when order is placed in buffer_in of the selected destination
                self.transported = self.env.event()
                self.order_log.append(["after_transport", self.id, round(self.env.now, 5), self.current_location.id])

            if self.get_next_step().type == 'sink':
                break

            self.order_log.append(["before_processing", self.id, round(self.env.now, 5), self.current_location.id])

            yield self.processed
            self.processed = self.env.event()

            self.order_log.append(["after_processing", self.id, round(self.env.now, 5), self.current_location.id])

        self.set_eop()
        self.statistics['stat_order_waiting'][self.id] = self.get_total_waiting_time()  # Calling this procedure updates the order waiting time statistics
        self.statistics['stat_total_waiting_time_orders'] += self.statistics['stat_order_waiting'][self.id]
        self.statistics['orders_done'].append(self)
        # print("ORDER_PRIORITY_MED : ", self.statistics['ORDER_PRIORITY_MED'])
        if self.order_priority == 3:
            self.statistics['ORDER_PRIORITY_LOW'] += 1
            self.statistics['stat_total_waiting_time_orders_low'] += self.statistics['stat_order_waiting'][self.id]
        elif self.order_priority == 2:
            self.statistics['ORDER_PRIORITY_MED'] += 1
            self.statistics['stat_total_waiting_time_orders_medium'] += self.statistics['stat_order_waiting'][self.id]
        else:
            self.statistics['ORDER_PRIORITY_HIGH'] += 1
            self.statistics['stat_total_waiting_time_orders_high'] += self.statistics['stat_order_waiting'][self.id]
        self.current_location = None