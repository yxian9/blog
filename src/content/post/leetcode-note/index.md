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



- 2395&#46; subarray with equal sum
```python
    def findSubarrays(self, nums: List[int]) -> bool:
        res = set()
        for i in range(1, len(nums)):
            cur_sum = nums[i] + nums[i-1]
            if cur_sum in res:
                return True
            res.add(cur_sum)
        return False
```
- 170&#46; Two Sum III - Data structure design
```js
class TwoSum {
    arr: Map<number, number>
    constructor() {
        this.arr = new Map<number, number>()
    }
    add(number: number): void {
        this.arr.set(number, (this.arr.get(number) || 0) + 1)
    }
    find(value: number): boolean {
        for (const [key, freq] of this.arr) {
            const diff = value - key
            if (diff === key) {
                if (freq > 1) return true
                // return freq >1 incorrect, pre-return
            } else {
                if (this.arr.has(diff)) return true
            }
        }
        return false
    }
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
## prefix sum
- 523&#46; Continuous Subarray Sum, the sum of the elements of the subarray is a multiple of k.
```python
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        pre = 0
        two_sum = {0: -1} ## 
        for i, v in enumerate(nums):
            pre += v
            rem  = pre % k
            if rem in two_sum:
                if i - two_sum[rem] >= 2:
                    return True
            else: ## need else to avoid overwrite first occurance
                two_sum[rem] = i
        return False
```

- 560&#46; Subarray Sum Equals K, the total number of continuous subarrays whose sum equals to k.
```python
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre = 0
        dic = {0: 1}
        res = 0
        for v in nums:
            pre += v
            diff = pre -k
            count += dic.get(diff, 0)
            dic[pre] = dic.get(pre, 0) + 1
        return count
# or
            if diff in dic:
                count += dic[diff]
            if pre in dic:
                dic[pre] += 1
            else:
                dic[pre] = 1
```
---
## interval
- 252&#46; Meeting Rooms I
```python
    def canAttendMeetings(self, intervals: List[List[int]) -> bool:
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False
        return True
```

- 253&#46; Meeting Rooms II
```python
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        up_down = []
        for start, end in intervals:
            up_down.append((start,  1))
            up_down.append((end, -1))
        up_down.sort()
        res, cur = 0
        for _, n in up_down:
            cur += n
            res = max(cur, res)
        return res 
```
```js
function minMeetingRooms(intervals: number[][]): number {
    const upDown: [number, number][] = []
    for (const [start, end] of intervals) {
        upDown.push([start, 1])
        upDown.push([end, -1])
    }
    upDown.sort((a, b) => a[0] - b[0] || a[1] - b[1])
    let [max, cur] = [0, 0]
    for (const [_, n] of upDown) {
        cur += n
        max = Math.max(cur, max)
    }
    return max

}
```
- 236&#46; Lowest Common Ancestor of a Binary Tree
```python
def lowestCommonAncestor(self, root, p, q):
    if root is None or root is p or root is q:
        return root
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```
```js
function lowestCommonAncestor(root: TreeNode | null, p: TreeNode | null, q: TreeNode | null): TreeNode | null {
    if(root === null || root === p || root === q) return root
    const l = lowestCommonAncestor(root.left, p, q)
    const r = lowestCommonAncestor(root.right, p, q)
    if(l === null) return r
    if(r === null) return l
    return root
}
```
- 98&#46; Validate Binary Search Tree
```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def dfs(node, l , r):
        if node is None:
            return True
        if not l < node.val < r: # start with negative condition
            return False
        return dfs(node.left, l, node.val) and dfs(node.right, node.val, r)
    return dfs(root, -inf, inf
```
```js
function isValidBST(root: TreeNode | null): boolean {
    function dfs(node: TreeNode | null, l: number, r: number){
        if(node === null) return true
        if(!( node.val > l && node.val < r)) return false
        return dfs(node.left, l, node.val) && dfs(node.right, node.val, r)
    }
    return dfs(root, -Infinity, Infinity)    
}
```
- 701&#46; Insert into a Binary Search Tree
```js
function insertIntoBST(root: TreeNode | null, val: number): TreeNode | null {
    if(!root) return new TreeNode(val)
    if( val > root.val){
        root.right = insertIntoBST(root.right, val)
    }else{
        root.left = insertIntoBST(root.left, val)
    }
    return root
}
```
---
## recursion
- 297&#46; Serialize and Deserialize Binary Tree
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        res = []
        def dfs(node: TreeNode):
            if node is None:
                res.append('x')
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(res)

    def deserialize(self, data):
        res = iter(data.split(','))
        def dfs(res):
            cur = next(res)
            if(cur == 'x'):
                return None
            node = TreeNode(int(cur))
            node.left = dfs(res)
            node.right = dfs(res)
            return node
        return dfs(res)
```
```js
 class TreeNode {
     val: number
     left: TreeNode | null
     right: TreeNode | null
     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
         this.val = (val===undefined ? 0 : val)
         this.left = (left===undefined ? null : left)
         this.right = (right===undefined ? null : right)
}
  
function serialize(root: TreeNode | null): string {
    function dfs(node: TreeNode | null) {
        if (!node) return 'x'
        return node.val.toString() + ',' + dfs(node.left) + ',' + dfs(node.right)
    }
    return dfs(root)
};

function deserialize(data: string): TreeNode | null {
    const res = data.split(',')[Symbol.iterator]()
    function dfs(res: IterableIterator<string>) {
        const cur = res.next().value
        if (cur === 'x') return null
        const node = new TreeNode(parseInt(cur))
        node.left = dfs(res)
        node.right = dfs(res)
        return node
    }

    return dfs(res)
}
```
---
## backtracking
- 22&#46; Generate Parentheses
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def dfs(s, l, r):
            if len(s) == 2 * n:
                res.append(s)
                return
            if l < n:
                dfs(s + '(', l + 1, r)
            if r < l:
                dfs(s + ')', l, r + 1)
        dfs('', 0, 0)
        return res
```
```js
function generateParenthesis(n: number): string[] {
    const res: string[] = []
    function dfs(path: string[], l: number, r: number) {
        if (path.length == 2 * n) {
            res.push(path.join(''))
            return
        }
        if (l < n) {
            path.push('(')
            dfs(path, l + 1, r)
            path.pop()
        }
        if (r < l) {
            path.push(')')
            dfs(path, l, r + 1)
            path.pop()
        }
    }
    dfs([], 0, 0)
    return res
}
```
- 698&#46; Partition to K Equal Sum Subsets
```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        def dfs(i):
            if i == len(nums):
                return True
            for j in range(k):
                if j > 0 and cur[j] == cur[j - 1]:
                    continue
                cur[j] += nums[i]
                if cur[j] <= s and dfs(i + 1):
                    return True
                cur[j] -= nums[i]
            return False

        s, mod = divmod(sum(nums), k)
        if mod:
            return False
        cur = [0] * k
        nums.sort(reverse=True)
        return dfs(0)
```
```js
function canPartitionKSubsets(nums: number[], k: number): boolean {
    const tot = nums.reduce((acc, cur) => acc + cur, 0)
    if (tot % k != 0) return false
    nums.sort((a, b) => a - b)
    const n = nums.length;
    const t = tot / k

    function dfs(idx: number, cur: number, cnt: number, vis: boolean[]): boolean {
        if (cnt == k) return true
        if (cur == t) return dfs(n - 1, 0, cnt + 1, vis)
        for (let i = idx; i >= 0; i--) {
            if (vis[i] || cur + nums[i] > t) continue
            vis[i] = true
            if (dfs(i - 1, cur + nums[i], cnt, vis)) return true
            vis[i] = false
            if (cur == 0) return false
        }
        return false
    }
    return dfs(n - 1, 0, 0, new Array<boolean>(n).fill(false))
};
```
---
## backtracking + dp
- 139&#46; Word Break
```js
function wordBreak(s: string, wordDict: string[]): boolean {
    const dict = new Set(wordDict);
    const dp = new Array(s.length + 1).fill(false);
    dp[0] = true; // Base case: empty string
    // string dp. 
    for (let j = 0; j <= s.length; j++) { 
        for (const item of wordDict) {
            if( item.length > j) continue
            if (dp[j - item.length] && dict.has(s.slice(j - item.length, j))) {
                dp[j] = true; 
                break
            }
        }
    }
    return dp[s.length]; 
}
```
```python
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        dp = [False for i in range(len(s) + 1)]
        dp[0] = True
        for i in range(len(s) + 1):
            for word in words:
                if i >= len(word):
                    if dp[i - len(word)] and s[i - len(word) : i] in words:
                        dp[i] = True
                        break
        return dp[len(s)
```
```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
  dic = set(wordDict)
  @cache
  def dfs(idx):
      if idx == len(s):
          return True
      res = false  
      for item in dic:
          if s[idx:].startswith(item):
              if dfs(idx + len(item)):
                  res = True
                  break
              res = dfs(idx + len(item)) # or
              if res:
                  break
      return res
  return dfs(0)
```
```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    dic = set(wordDict)
    @cache
    def dfs(idx):
        if idx == len(s):
            return True
        res = False
        for item in dic:
            pre = s[idx : idx + len(item)]
            if pre in dic:
                if dfs(idx + len(item)):
                    res = True
                    break
                res = dfs(idx + len(item))
                if res:
                    break
        return res
    return dfs(0)
```
- 91&#46; Decode Ways
```js
var numDecodings = function(s) {
    const dict = Array.from(Array(26).keys(), idx => (idx +1).toString())
    // const set = new Set(dict)
    const map = new Map()
    function dfs(idx) {
        if(idx > s.length) return 0 
        if(idx === s.length) return 1
        if(map.has(idx)) return map.get(idx)
        let res = 0
        for (let item of dict) {
            if((idx + item.length) > s.length) continue
             const pre = s.slice(idx, idx + item.length)
            
            if( set.has(pre)){ // !!!
                res += dfs(idx + item.length);
            }
            if (pre === item) { // not set.has(pre)
                res += dfs(idx + item.length);
            }
        }
        map.set(idx, res)
        return res
    }
    return dfs(0)
}
```
```js
