import turtle as t
import math
import time

class Drones():
    def __init__(self, n, starting_points, final_points):
        self.done = False
        self.reward = 0

        self.win = t.Screen()
        self.win.title('Drones')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)
        self.win.setworldcoordinates(0,0,100,100)

        self.max_velocity = 5
        self.min_velocity = 1

        self.drones = list()
        for i in range(n):
            temp_drone = t.Turtle()
            temp_drone.speed(0)
            temp_drone.shape('circle')
            temp_drone.color('red')
            temp_drone.penup() 
            temp_drone.goto(starting_points[i]["x"], starting_points[i]["y"])
            temp_drone.velocity = 3
            temp_drone.destx = final_points[i]["x"]
            temp_drone.desty = final_points[i]["y"]
            self.drones.append(temp_drone) 

        for i in range(n):
            heading_angle = math.degrees(math.atan2((self.drones[i].desty - self.drones[i].ycor()), self.drones[i].destx - self.drones[i].xcor()))
            self.drones[i].setheading(heading_angle)

    def increase_velocity(self, drone_id):
        self.drones[drone_id].velocity += 2
        if self.drones[drone_id].velocity > self.max_velocity:
            self.drones[drone_id].velocity = self.max_velocity
        
    def decrease_velocity(self, drone_id):
        self.drones[drone_id].velocity -= 2
        if self.drones[drone_id].velocity < self.min_velocity:
            self.drones[drone_id].velocity = self.min_velocity
    
    def check_formation_complete(self, n):
        for i in range(n):
            if (round(self.drones[i].xcor(), 1) != self.drones[i].destx) or (round(self.drones[i].ycor(), 1) != self.drones[i].desty):
                return False
        return True

    def distance_between_drones(self, i, j):
        interdrone_distance_i_j= math.sqrt(((self.drones[i].xcor() - self.drones[j].xcor()) ** 2) + ((self.drones[i].ycor() - self.drones[j].ycor()) ** 2))
        return interdrone_distance_i_j

    def check_collision(self, n):
        for i in range(n):
            for j in range(i+1, n):
                if self.distance_between_drones(i,j) <= 2:
                    return True
        return False
    
    def check_out_of_frame(self, n):
        for i in range(n):
            if (self.drones[i].xcor() > 100) or (self.drones[i].ycor() > 100):
                return True

            if (self.drones[i].xcor() < -100) or (self.drones[i].ycor() < -100):
                return True

    def run_frame(self, n):
        self.win.update()

        for i in range(n):
            rem_velocity = round(math.sqrt(((self.drones[i].xcor() - self.drones[i].destx) ** 2) + ((self.drones[i].ycor() - self.drones[i].desty) ** 2)), 1)
            if rem_velocity < self.drones[i].velocity:
                self.drones[i].velocity = rem_velocity
            self.drones[i].forward(self.drones[i].velocity)
            # print("Drone ", i, " coordinates:", self.drones[i].xcor(), self.drones[i].ycor())

        if self.check_collision(n):
            self.reward -= 3
            self.done = True

        if self.check_formation_complete(n):
            self.reward += 3
            self.done = True
        
        if self.check_out_of_frame(n):
            self.reward -= 3
            self.done = True
    
    def get_current_state(self, n):
        state = list()
        for i in range(n):
            state.append(self.drones[i].xcor())
            state.append(self.drones[i].ycor())
        
        for i in range(n):
            state.append(self.drones[i].velocity)

        return state

    def step(self, actions, n):
        critical_drones = self.find_critical_drones(n)
        self.reward = 0
        self.done = 0
        for i in critical_drones:
            if actions[i] == 0:
                self.increase_velocity(i)
                self.reward -= 0.1


            if actions[i] == 2:
                self.decrease_velocity(i)
                self.reward -= 0.1


        self.run_frame(n)
        self.reward -= 0.01


        state = self.get_current_state(n)
        return self.reward, state, self.done

    def reset(self, n, starting_points, final_points):
        for i in range(n):
            self.drones[i].goto(starting_points[i]["x"], starting_points[i]["y"])
            self.drones[i].velocity = 3
            self.drones[i].destx = final_points[i]["x"]
            self.drones[i].desty = final_points[i]["y"]

        state = self.get_current_state(n)
        return state
    
    def find_critical_drones(self, n):
        critical_drones = set()
        for i in range(n):
            for j in range(i+1, n):
                if self.distance_between_drones(i,j) <= 5:
                    critical_drones.add(i)
                    critical_drones.add(j)
        return critical_drones


#Sample code to test the working of the model

if __name__ == "__main__":
    starting_points = list()
    final_points = list()
    
    with open('/home/saran/Desktop/projects/rl/drones/Input_data/inp_1.txt') as e:
        n = int(e.readline())
        for i in range(n):
            line = e.readline()
            tempx, tempy = line.split(" ")
            x = int(tempx)
            y = int(tempy)
            point = dict({"x": x, "y":y})
            starting_points.append(point)
        for i in range(n):
            line = e.readline()
            tempx, tempy = line.split(" ")
            x = int(tempx)
            y = int(tempy)
            point = dict({"x": x, "y":y})
            final_points.append(point)
    
    env = Drones(n, starting_points, final_points)
    env.increase_velocity(2)
    # env.increase_velocity(5)

    while env.done != True:
        env.run_frame(n) 
        time.sleep(0.1)

    print("Reward =", env.reward)