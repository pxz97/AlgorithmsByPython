class solution:
    def NumberOf1Between1AndN_Solution(self, n):
        """
        整数中1出现的次数（从1到n整数中1出现的次数）
        :param n: int, 给定的整数n
        :return: int, 1~n的十进制中1出现的次数
        """
        cur, digit = n % 10, 1
        high, low = n // 10, 0
        count = 0
        while high != 0 or cur != 0:
            if cur == 0:
                count += high * digit
            elif cur == 1:
                count += high * digit + low + 1
            else:
                count += high * digit + digit
            low += cur * digit
            digit *= 10
            cur = high % 10
            high //= 10
        return count
