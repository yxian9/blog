---
title: "Leetcode note"
description: "code gymnastics"
publishDate: "22 Mar 2024"
tags: ["programming", "algo", "data structure"]
---


## Two pointer
### binary search 
- 34&#46; Find First and Last Position of Element in Sorted Array

```python
class Solution:
    def bs(self, arr: list[int], target: int, first: bool) -> int:
        l, r = 0, len(arr) - 1
        idx = -1
        while l <= r:
            mid = (l + r) // 2
            if arr[mid] == target:
                idx = mid
                if first: # find first
                    r = mid - 1
                else: # fird last
                    l = mid + 1
            elif arr[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return idx
        
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        i1 = self.bs(nums, target, True)
        if i1 == -1:
            return [-1, -1]
        i2 = self.bs(nums, target, False)
        return [i1, i2]
```
- 153&#46; Find Minimum in Rotated Sorted Array, (fissible function)
```python
    def findMin(self, nums: List[int]) -> int:
        l, r, idx = 0, len(nums) -1, -1
        while l <=r :
            mid = (l +r) //2
            if(nums[mid] <= nums[-1]): # may equal
                idx = mid
                r = mid -1
            else:
                l = mid +1
        return nums[idx]
```
- 852&#46; Peak Index in a Mountain Array. [0,2,1,0] return idx 1
```python
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        l, r, idx = 0, len(arr) - 1, -1
        while l <= r:
            mid = (l + r) //2
            if mid == len(arr) -1 or arr[mid] > arr[mid + 1]:
              # mid = len(arr) -1, is consider true
              # to fullfill the monotonous
                idx = mid
                r = mid - 1 
            else:
                l = mid + 1
        return idx
  ## or
        while l <= r:
            mid = (l + r) //2
            if mid ==0 or arr[mid - 1] < arr[mid]:
                idx = mid
                l = mid + 1 
            else:
                r = mid - 1
        return idx
```

### window 
- 346&#46; Given a stream of integers and a window size, calculate the moving average. fixed window
```python
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.que = collections.deque()
        self.sum = 0

    def next(self, val: int) -> float:
        if len(self.que) < self.size:
            # self.que.append(val)
            self.sum += val
        else:
            self.sum += val - self.que.popleft()
            # self.que.append(val)
        self.que.append(val)
        return self.sum / len(self.que)

# movingAverage = new MovingAverage(3);
# movingAverage.next(1); // return 1.0 = 1 / 1
# movingAverage.next(10); // return 5.5 = (1 + 10) / 2
```
-- since we are not care about the order within the window, only need to keep track the last item. use single pointer.

```python
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.que = [0] * size # need initial full size, or 
        self.sum = 0
        self.count = 0

    def next(self, val: int) -> float:
        idx = self.count % self.size ## last item 
        self.sum += val - self.que[idx]
        self.que[idx] = val ## not push, require preset length
        self.count += 1
        return self.sum / min(self.count, self.size)
```
- 300&#46; Longest Increasing Subsequence  
construct the growing window, maintain the first larger element.

```python
class Solution:
    def first_large(self, arr: list[int], target: int) ->int:
        l, r, idx = 0, len(arr) -1, -1
        while l <= r:
            mid = ( l + r) //2
            if arr[mid] >= target:
                idx = mid
                r = mid -1
            else:
                l = mid +1 
        return idx

    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []
        for i in nums:
            idx = self.first_large(res, i)
            if idx == -1:
                res.append(i)
            else:
                res[idx] = i
        return len(res)
```

- 2099&#46; find a subsequence of nums
 of length k that has the largest sum.
 ```python
 class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        idxs = list(range(len(nums)))
        idxs.sort(key=lambda i: nums[i])
        sub_idxs = idxs[-k:]
        sub_idxs.sort()
        return [nums[i] for i in sub_idxs]

```
-- or greedy, keep the min idx for each round.
```python
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        res = []
        for val in nums:
            if len(res) < k:
                res.append(val)
                continue
            min_idx, min_v = 0, res[0]
            for i, v in enumerate(res):
                if v < min_v:
                    min_idx, min_v = i, v

            if val > min_v:
                res[min_idx] = va
        final = []
        res = Counter(res)
        for i in nums:
            # if i in res:j
                # final.append(i)
                # res.remove(i)
            # for j, v in enumerate(res):
            #     if i == v:
            #         final.append(i)
            #         res[j] = None
            #         break ## need break
            if res[i] > 0:
                final.append(i)
                res[i] -= 1

        return final
```

---



