class Solution:
    def isNumeric(self, s):
        states = [
            {" ": 0, "s": 1, "d": 2, ".": 4},   # 0. 开始的空格" "
            {"d": 2, ".": 4},                   # 1. 幂符号前的正负号
            {"d": 2, ".": 3, "e": 5, " ": 8},   # 2. 幂符号前的整数部分
            {"d": 3, "e": 5, " ": 8},           # 3. 幂符号前的小数点、小数部分数字
            {"d": 3},                           # 4. 当小数点前为空格时，小数点、小数点后的数字
            {"s": 6, "d": 7},                   # 5. 幂符号
            {"d": 7},                           # 6. 幂符号后的正负号
            {"d": 7, " ": 8},                   # 7. 幂符号后的数字
            {" ": 8}                            # 8. 结尾的空格
        ]

        p = 0

        for c in s:
            # 判断当前字符类型，并赋值给t
            if "0" <= c <= "9":
                t = "d"
            elif c in "+-":
                t = "s"
            elif c in "eE":
                t = "e"
            elif c in ". ":
                t = c
            else:
                t = "?"
            if t not in states[p]:  # 判断：当前字符类型是否在【前一个字符所处状态】的【可转移状态】里
                return False
            p = states[p][t]  # 当前字符所处的状态（0~8），供下一个字符做判断（if t not in states[p]:）
        return p in (2, 3, 7, 8)
