class remoted_car:    
    def __init__(self):
        """
        do some inintialization
        """ 
        self.__command_left = 1
        self.__command_right = 2 
        self.__command_head = 3
        self.__command_stop = 4   
        self.__command=4   
        return  
    def get_c(self, dist):
        if dist<0.5 and dist>0.2:
            self.__command= self.__command_left
        elif dist>=0.5:
            self.__command= self.__command_head
        else:
            self.__command = self.__command_stop
    def turn_left(self):
        print("turn left")
        return
    def turn_right(self):
        print("turn right")
        return
    def head(self):
        print("head")
        return
    def stop(self):
        print("stop") 
        return
    def move(self):
        if self.__command==self.__command_head:
            self.head()
        elif self.__command==self.__command_stop:    
            self.stop()
        elif self.__command==self.__command_left:
            self.turn_left()
        return
    
