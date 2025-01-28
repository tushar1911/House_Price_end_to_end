from abc import ABC,abstractmethod

class Coffee(ABC):
    @abstractmethod
    def prepare(self):
        pass

class Espresso(Coffee):
    def prepare(self):
        return "Preparing a rich and strong Espresso."
    
class Latte(Coffee):
    def prepare(self):
        return "Preparing a smooth and creamy Latte."
    

class CoffeeMachine:
    def make_coffee(self, coffee_type):
        if coffee_type=="Espresso":
            return Espresso().prepare()
        elif coffee_type=="Latte":
            return Latte().prepare()
        else:
            return "Unknown coffee type!"
    
if __name__=="__main__":
    machine=CoffeeMachine()
    
    coffee=machine.make_coffee("Espresso")
    print(coffee)
    
    cofee=machine.make_coffee("Latte")
    print(coffee)