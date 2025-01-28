from abc import ABC,abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def pay(self,amount):
        pass
    
class CreditCardPayment(PaymentMethod):
    def pay(self,amount):
        return f"Paying {amount} using Credit Card"
    
class PaypalPayment(PaymentMethod):
    def pay(self, amount):
        return f"Paying {amount} using Paypal"
    
class ShoppingCart:
    def __init__(self, payment_method: PaymentMethod):
        self.payment_method= payment_method
    
    def checkout(self,amount):
        return self.payment_method.pay(amount)

if __name__=="__main__":
    cart=ShoppingCart(CreditCardPayment())
    print(cart.checkout(100))
    
    cart=ShoppingCart(PaypalPayment())
    print(cart.checkout(100))
    