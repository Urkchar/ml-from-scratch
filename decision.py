class Decision():
    def __init__(self, left_operand, right_operand, operator):
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.operator = operator

    def __call__(self):
        return self.operator(self.left_operand, self.right_operand)
