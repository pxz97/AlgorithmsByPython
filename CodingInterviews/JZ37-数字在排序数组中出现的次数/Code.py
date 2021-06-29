class Solution:
    def GetNumberOfK(self, data, k):
        if self.GetFirstOfK(data, k) == -1 and self.GetLastOfK(data, k) == -1:
            return 0
        return self.GetLastOfK(data, k) - self.GetFirstOfK(data, k) + 1

    def GetFirstOfK(self, data, k):
        left, right = 0, len(data) - 1
        while left <= right:
            mid = (left + right) // 2
            if data[mid] < k:
                left = mid + 1
            elif data[mid] > k:
                right = mid - 1
            else:
                if left == mid or data[mid - 1] != k:
                    return mid
                else:
                    right = mid - 1
        return -1

    def GetLastOfK(self, data, k):
        left, right = 0, len(data) - 1
        while left <= right:
            mid = (left + right) // 2
            if data[mid] < k:
                left = mid + 1
            elif data[mid] > k:
                right = mid - 1
            else:
                if right == mid or data[mid + 1] != k:
                    return mid
                else:
                    left = mid + 1
        return -1
