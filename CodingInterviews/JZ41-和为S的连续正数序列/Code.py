class Solution:
    def FindContinuousSequence(self, tsum):
        head, tail, sum_ = 1, 2, 3
        res = []
        while head < tail:
            if sum_ < tsum:
                tail += 1
                sum_ += tail
            elif sum_ > tsum:
                sum_ -= head
                head += 1
            else:
                res.append(list(range(head, tail + 1)))
                sum_ -= head
                head += 1
        return res
