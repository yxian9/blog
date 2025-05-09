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
```python
def lengthOfLIS(self, nums: List[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for right in range(1, n):
        for left in range(right):
            if nums[right] > nums[left]:
                if dp[right] < dp[left] + 1:
                    dp[right] = dp[left] + 1
# no dp[right] += 1 or dp[left] += 1
# use dp[left] + 1 to update dp[right]
# if right > left, then choose one of dp[left] to add one(to get larger res)
    return max(dp)
```

- 674&#46; Longest Continuous Increasing Subsequence
```python
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        # res = 1
        # cur_length = 1
        # for right in range(1, len(nums)):
        #     if nums[right] > nums[right-1]:
        #         cur_length  += 1
        #         res = max(res, cur_length)
        #     else:
        #         cur_length = 1
        # return res

        dp = [1] * len(nums)
        for right in range(1, len(nums)):
            if nums[right] > nums[right-1]:
                dp[right] =  dp[right-1] + 1
            #     res = max(res, cur_length)
            # else:
            #     cur_length = 1
        return max(dp)
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
                res[min_idx] = v
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
 interval function
```python
def overlap(a, b):
    return not (b[1] < a[0] or a[1] < b[0]) 
```
```js
const overlap = (a, b) => !(b[1] < a[0] || a[1] < b[0]);
```

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
    return dfs(root, -inf, inf)
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
## linked list

- 203&#46; Remove Linked List Elements
```python
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        pre, cur = dummy, head
        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        return dummy.next
        ## or
        pre = dummy
        while pre.next: 
## no need check pre && pre.next. pre will never be None. 
## while condition is checked first. dummy garantee pre.next exist 
            if pre.next.val == val:
                pre.next = pre.next.next
            else:
                pre = pre.next
        return dummy.next

```

---
## recursion
- 257&#46; Binary Tree Paths
```python
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        res = []
        def dfs(node, path):
            if node is None:
                return
            path += str(node.val)
            if node.left is None and node.right is None:
                res.append(path)
                return
            path += '->'
            dfs(node.left, path)
            dfs(node.right, path)
        dfs(root, '')
        return res
```
```python
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        res = []

        def dfs(node, path):
            if node is None:
                return
            path.append(str(node.val))

            if node.left is None and node.right is None:
                res.append("->".join(path))
                # return
            choices = []
            if node.left:
                choices.append(node.left)
            if node.right:
                choices.append(node.right)
            for choice in choices:
                dfs(choice, path)
                path.pop()
            #or pop here but need to delete return in the append
            path.pop()

        dfs(root, [])
        return res
```



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

- 89&#46; Gray Code
```python
    def grayCode(self, n: int) -> List[int]:
        length, visited = 1 << n , [False] * length
        path = [0]
        visited[0] = True

        def dfs(code):
            if len(path) == length:
                return True
            for i in range(n):
                new_code = code ^ (1 << i)

                if visited[new_code]:
                    continue
                path.append(new_code)
                visited[new_code] = True
                if dfs(new_code):
                    break ## or return True
                visited[new_code] = False
                path.pop()
            return True ## no need if pre return
         
        dfs(0)
        return path
```

---
## backtracking + dp
- 124&#46; Binary Tree Maximum Path Sum
```python
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = -inf
        def dfs(node):
            nonlocal res
            if node is None:
                return 0
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            res = max(res, node.val + left + right)
            return node.val + max(left, right)
        dfs(root)
        return res
        # or 
        def dfs(node):
            if node is None:
                return 0
            l = dfs(node.left)
            r = dfs(node.right)
            nonlocal ans
            ans = max(ans, l + r + node.val)
            return max(max(l, r) + node.val, 0)
        dfs(root)
        return ans
```

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
function numDecodings(s) {
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
- 322&#46; Coin Change
```js
function coinChange(coins: number[], amount: number): number {
    const dp = new Array(amount + 1).fill(-1)
    dp[0] = 0
    for(const item of coins){
        for( let j = 0; j <= amount; j++){
            if(  j >= item && dp[j - item] !== -1 ) {
                if(dp[j] !== -1){
                    dp[j] = Math.min(dp[j], dp[j - item] + 1)  
                } else{
                    dp[j] = dp[j - item ] +1
                }
            }
        }
    }
    return dp[amount] 
}
```
```python
def coinChange(self, coins: List[int], amount: int) -> int:
    # dp = [0] + [inf] * amount
    dp = [0] + [amount + 1] * amount
    for item in coins:
        for j in range(item, amount + 1):
            if dp[j] > dp[j - item] + 1:
                dp[j] = dp[j - item] + 1
    if dp[amount] == amount + 1:
        return -1
    return dp[amount]
```
```ts
function coinChange(coins: number[], amount: number): number {
    const dp = new Array(amount + 1).fill(amount + 1)
    dp[0] = 0
    for (const item of coins) {
        for (let j = item; j <= amount; j++) {
          // dp[j] could be uninitial or a larger one
            if (dp[j] > dp[j - item] + 1) { 
                dp[j] = dp[j - item] + 1
            }
        }
    }
    if (dp[amount] === amount + 1) return -1
    return dp[amount]
}
```
```ts
function coinChange(coins: number[], amount: number): number {
    const memo = new Map<number,number>()

    function dfs(depth: number, total: number){
        if(memo.has(total)) return memo.get(total)
        if(total === amount) {
            return depth // can not memo here 
        } 
        if(total > amount){
            return -1
        }
        let ans = Infinity
        for(const item of coins){
            const res = dfs(depth + 1, total + item)
            if( res !== -1 && ans > res){
                ans = res
            }
        }
        memo.set(total,ans) 
        // memo is give current total, additional coins need
        // not mini coins need to get current total.
        return ans
    }
    const ans = dfs(0,0)

    return ans ===Infinity ? -1: ans
    
};
```
- 494&#46; Target Sum
```js
function findTargetSumWays(nums: number[], target: number): number {
    const total = nums.reduce( (acc, cur) => acc + cur, 0)
    if( Math.abs(total) < Math.abs(target)) return 0
    if( (total + target) %2 !== 0) return 0
    const expected = ( total + target) / 2
    const dp = new Array(expected +1).fill(0)
    dp[0] = 1
    for( const item of nums) {
        for( let j = expected; j >= item; j--){ // >= not =>
            dp[j] = dp[j] + dp[j - item]
        }
    }
    return dp[expected];
}
```
```python
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > abs(total) :
            return 0
        if( (target + total) %2 == 1):
            return 0
        expected =  (target + total) // 2

        dp = [1] + [0] * expected
        for item in nums:
            for c in range( expected, item -1, -1):
                dp[c] += dp[ c - item]
        return dp[expected]
```


- 300&#46; Longest Increasing Subsequence
```js
function lengthOfLIS(nums: number[]): number {
    if(nums.length === 0) return 0
    const dp = new Array(nums.length).fill(1)
    for(let i= 0; i < nums.length; i++){
        const lastItem = nums[i]
        for(let j = 0; j < i; j++){
            if(lastItem > nums[j]){
                if(dp[j] + 1 > dp[i]){
                    dp[i] = dp[j] + 1
                }
                // dp[i] = Math.max(dp[i], dp[j] + 1) // j always less then i, or i-1 
            }
        }
    }
    return Math.max(...dp)
}
```
- 368&#46; Largest Divisible Subset
```js
function largestDivisibleSubset(nums: number[]): number[] {
    nums.sort((a,b) => a - b)
    const dp = new Array(nums.length).fill(1)
    const prev = new Array(nums.length).fill(-1)
    for (let i = 0; i < nums.length; i++) {
        const lastItem = nums[i]
        for (let j = 0; j < i; j++) {
            if (lastItem % nums[j] === 0) {
                if (dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1
                    prev[i] = j //bookkeeping
                }
            }
        }
    }
    let maxIdx = dp.indexOf(Math.max(...dp))
    const res = []
    while( maxIdx !== -1){
        res.push(nums[maxIdx])
        maxIdx = prev[maxIdx]
    }
    return res
}
```
- 62&#46; Unique Paths
```js
var uniquePaths = function (m, n) {
    const dp = Array.from({ length: m }, () => new Array(n).fill(0))
    for (let i = 0; i < m; i++) {
        dp[i][0] = 1
    }
    for (let j = 0; j < n; j++) {
        dp[0][j] = 1
    }
    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        }
    }
    return dp[m - 1][n - 1]
}
```
```go
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := range(dp) {
		dp[i] = make([]int, n)
	}
	for i := range(dp) {
		for j := range(dp[0]) {
			if i == 0 || j == 0 {
				dp[i][j] = 1
				continue
			}
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}
```
- 63&#46; Unique Paths II
```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}

	for i := range dp {
		if obstacleGrid[i][0] == 1 {
			break
		}
		dp[i][0] = 1
	}
	for j := range dp[0] {
		if obstacleGrid[0][j] == 1 {
			break
		}
		dp[0][j] = 1
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] == 0 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}
```

- 64&#46; Minimum Path Sum
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = grid[i][j]

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # need to avoid update [0][0] item
                if i == 0 and j > 0:
                    dp[i][j] += dp[i][j - 1]
                elif j == 0 and i > 0:
                    dp[i][j] += dp[i - 1][j]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]
# or 
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    dp[i][j] = grid[i][j]
                elif i == 0:
                    dp[i][j] = grid[i][j] + dp[i][j - 1]  ## set and overwrite same time
                elif j == 0:
                    dp[i][j] = grid[i][j] + dp[i - 1][j] 
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]
```

- 198&#46; House Robber
```python
    def rob(self, nums: List[int]) -> int:
        dp = [0] + [0] * len(nums)
        dp[1] = nums[0] # the nums and dp has offset 1
        for i in range(2, len(nums) + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        return dp[len(nums)]
```


---
## stack
- 227&#46; Basic Calculator II
```python
    def calculate(self, s: str) -> int:
        stack, sign, num = [], "+", 0
        for i in range(len(s) + 1):
            if i < len(s):
                c = s[i]
                if c == " ":
                    continue
                if c.isdigit():
                    num = 10 * num + int(c)

            if (not c.isdigit()) or i == len(s):
                match sign:
                    case "+":
                        stack.append(num)
                    case "-":
                        stack.append(-num)
                    case "*":
                        stack[-1] = stack[-1] * num
                    case "/":
                        stack[-1] = int(stack[-1] / num)
                num = 0
                sign = c
        return sum(stack)
```

- 239&#46; Sliding Window Maximum
```python
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        que = deque()
        for i, v in enumerate(nums):
            while que and nums[que[-1]] < v:
                que.pop()
            que.append(i)
            if i - que[0] >= k:
                que.popleft()
            if i >= k - 1:
                res.append(nums[que[0]])
        return res
```
```js
function maxSlidingWindow(nums: number[], k: number): number[] {
    const ans: number[] = []
    const queue: number[] = []
    let j = 0
    for(let i =0; i< nums.length; i++){
        while( queue.length >j && nums[i] > queue[queue.length-1]  ){
            queue.pop()
        }
        queue.push(nums[i])
        if( i >= k-1){
            ans.push(queue[j])
            if(nums[i - k +1] === queue[j]){
                j++  // queue.shift()
            }
        }
    }
   return ans 
}
```
- 416&#46; Partition Equal Subset Sum
```js
function canPartition(nums: number[]): boolean {
    const total = nums.reduce((a, b) => a + b, 0)
    if (total % 2 === 1) return false
    const target = total >> 1
    const dp = new Array(target + 1).fill(false)
    dp[0] = true
    for (const item of nums) {
        for (let j = target; j >= item; j--) {
            if (dp[j - item] ) {
                dp[j] = true
            }
            // or 
            if (dp[j - item] + item > dp[j]) {
                dp[j] = dp[j - item] + item
            }
            //or 
            dp[j] = Math.max(dp[j] , dp[j- item] + item)
        }
    }
    return dp[target] 
}
```
```python
def canPartition(self, nums: List[int]) -> bool:
    @cache
    def dfs(start, cum):
        if cum == target:
            return True
        for i in range(start, len(nums)):
            cur = cum + nums[i]
            if cur <= target and dfs(i + 1, cur):
                return True
        return False
    return dfs(0, 0)
```
```python
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        target = total >> 1
        dp = [True] + [False] * target
        for item in nums:
            for j in range(target, item - 1, -1):
                if dp[j - item]:
                    dp[j] = True
        return dp[target]
``` 
----
## bfs
- 199&#46; Binary Tree Right Side View
```python
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res = []
        que = deque([root])
        while que:
            size = len(que)
            for i in range(size):
                node = que.popleft()
                if i == size - 1:
                    res.append(node.val)
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
        ## or
        while queue:
            res.append(queue[0].val)
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)
        return res
```
```js
function rightSideView(root: TreeNode | null): number[] {
    const res : number[] = []
    function dfs(node: TreeNode | null, depth: number){
        if(!node) return
        if(depth === res.length){
            res.push(node.val)
        }
        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1) // res is mutated. when go to left, res is increased.
    }
    dfs(root, 0)
    return res
}
```

- 111&#46; Minimum Depth of Binary Tree
```python
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        que, depth = deque([root]), 0
        while que:
            depth += 1
            for _ in range(len(que)):
                node = que.popleft()
                if not node.left and not node.right:
                    return depth
                if node.left:
                    que.append(node.left)
                if node.right:
                    que.append(node.right)
        # return depth
```
```python
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if root.left and root.right:
            return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
        elif root.left:
            return 1 + self.minDepth(root.left)
        elif root.right:
            return 1 + self.minDepth(root.right)
        else:
            return 
```
```js
var minDepth = function (root) {
    if (!root) return 0
    const l = minDepth(root.left)
    const r = minDepth(root.right)
    // if (l === 0) return minDepth(root.right)
    // if (r === 0) return minDepth(root.left)
    if( l === 0 && r === 0 ){// must check this first
        return 1
    }else if(r === 0){
        return l + 1
    }else if(l === 0){
       return r + 1
    }else{
         return Math.min(l, r) + 1
    }
}
var minDepth = function (root) {
    if (!root) return 0
    const l = minDepth(root.left)
    const r = minDepth(root.right)
    if (l > 0 && r > 0) {
        return Math.min(l, r) + 1
    } else if (r === 0) {
        return l + 1
    } else if (l === 0) {
        return r + 1
    } else {
        return 1 // base case to build up, interact with 0
    }
}

var minDepth = function (root) {
    let res = Infinity
    function dfs(node, depth) {
        if (!node) return
        depth++
        if (!node.left && !node.right) {
            res = Math.min(res, depth )
        }
        dfs(node.left, depth )
        dfs(node.right, depth )
    }
    dfs(root, 0)
    return res === Infinity ? 0: res
}
```

---
##graph
- 133&#46; Clone Graph
```python
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        visited = {}
        def dfs(node):
            if node in visited:
                return visited[node]
            clone = Node(node.val, [])
            visited[node] = clone
            for n in node.neighbors:
                clone.neighbors.append(dfs(n))
            return clone
        return dfs(node)
```
- 1197&#46; Minimum Knight Moves
```js
function minKnightMoves(x: number, y: number): number {
    let res = 0
    const seen = new Set<string>('0,0')
    let queue = new Array<[number, number]>([0 ,0])
    while( queue.length > 0){
        const q = []
        let len = queue.length
        while(len--){
            const [r, c] = queue.pop()
            if (r === x && c === y) return res
            for( const [dx, dy] of move){
                const item : [number, number] = [r + dx, c + dy]
                const id = `${r + dx},${c + dy}`
               if(seen.has(id)) continue 
               seen.add(id)
               q.push(item)
            }
        }
        queue = q
        res++
    }
    return res
```
- 286&#46; Walls and Gates
```python
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        if not rooms:
            return
        m, n = len(rooms), len(rooms[0])
        que = deque()
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    que.append((i, j))
        while que:
            x, y = que.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == 2147483647:
                    rooms[nx][ny] = rooms[x][y] + 1
                    que.append((nx, ny))
```
```js
function wallsAndGates(rooms: number[][]): void {
    const m = rooms.length
    const n = rooms[0].length
    const que: number[][] = []
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (rooms[i][j] === 0) {
                que.push([i, j])
            }
        }
    }
    const dir = [[0, 1],[1, 0], [0, -1], [-1, 0]]
    while (que.length) {
        let len = que.length  // not necessary
        while (len--) { // not necessary
            const [r, c] = que.shift()
            for (const [dx, dy] of dir) {
                const newX = r + dx
                const newY = c + dy
                if (newX < 0 || newX > m - 1 || newY < 0 || newY > n - 1) continue
                if (rooms[newX][newY] === 2147483647) {
                    rooms[newX][newY] = rooms[r][c] + 1
                    que.push([newX, newY])
                }
            }
        }
    }
};
```


