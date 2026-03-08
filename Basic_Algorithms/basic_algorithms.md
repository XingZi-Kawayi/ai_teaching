# 春招 AI 应用开发/Python 后端 手撕代码必背 30 题

## 使用说明
本文档整理了 AI 应用开发/Python 后端岗位春招中最常考的 30 道算法题，全部采用极简可默写的 Python 代码实现。建议按以下节奏复习：
- **Day1**：数组+字符串+链表（10 题）
- **Day2**：二叉树+二分（8 题）
- **Day3**：DP+栈+设计（9 题）
- **Day4**：全部二刷 + 闭眼默写

---

## 一、数组 & 字符串（考得最多）

### 1. 两数之和
```python
def two_sum(nums, target):
    d = {}
    for i, num in enumerate(nums):
        if target - num in d:
            return [d[target-num], i]
        d[num] = i
```

### 2. 无重复字符的最长子串
```python
def length_of_longest_substring(s):
    d = {}
    left = 0
    res = 0
    for right, c in enumerate(s):
        if c in d and d[c] >= left:
            left = d[c] + 1
        d[c] = right
        res = max(res, right - left + 1)
    return res
```

### 3. 有效括号
```python
def is_valid(s):
    stack = []
    dic = {')':'(', ']':'[', '}':'{'}
    for c in s:
        if c not in dic:
            stack.append(c)
        else:
            if not stack or stack.pop() != dic[c]:
                return False
    return not stack
```

### 4. 最长回文子串（中心扩散极简版）
```python
def longest_palindrome(s):
    def helper(l, r):
        while l >=0 and r < len(s) and s[l]==s[r]:
            l -=1
            r +=1
        return s[l+1:r]
    res = ''
    for i in range(len(s)):
        res = max(res, helper(i,i), helper(i,i+1), key=len)
    return res
```

---

## 二、链表（必须闭眼写）

### 5. 反转链表
```python
def reverse_list(head):
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre
```

### 6. 合并两个有序链表
```python
def merge_two_lists(l1, l2):
    dummy = cur = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

### 7. 环形链表
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### 8. 删除倒数第 N 个节点
```python
def remove_nth_from_end(head, n):
    dummy = ListNode(next=head)
    fast = slow = dummy
    for _ in range(n+1):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next
```

---

## 三、二叉树（AI 岗高频）

### 9. 二叉树最大深度
```python
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### 10. 层序遍历
```python
def level_order(root):
    if not root: return []
    from collections import deque
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

### 11. 对称二叉树
```python
def is_symmetric(root):
    def dfs(l, r):
        if not l and not r: return True
        if not l or not r: return False
        return l.val == r.val and dfs(l.left, r.right) and dfs(l.right, r.left)
    return dfs(root.left, root.right)
```

### 12. 最近公共祖先
```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right: return root
    return left or right
```

---

## 四、二分查找（背模板）

### 13. 二分查找（标准）
```python
def search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target:
            return m
        elif nums[m] < target:
            l = m + 1
        else:
            r = m -1
    return -1
```

### 14. 搜索旋转排序数组
```python
def search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l+r)//2
        if nums[m] == target: return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]:
                r = m-1
            else:
                l = m+1
        else:
            if nums[m] < target <= nums[r]:
                l = m+1
            else:
                r = m-1
    return -1
```

---

## 五、动态规划（只背最简单 5 道）

### 15. 爬楼梯
```python
def climb_stairs(n):
    a, b = 1, 1
    for _ in range(n-1):
        a, b = b, a+b
    return b
```

### 16. 打家劫舍
```python
def rob(nums):
    a = b = 0
    for num in nums:
        a, b = b, max(b, a+num)
    return b
```

### 17. 最长递增子序列
```python
def length_of_lis(nums):
    if not nums: return 0
    dp = [1]*len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)
```

### 18. 最长公共子序列
```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] +1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### 19. 零钱兑换
```python
def coin_change(coins, amount):
    dp = [float('inf')]*(amount+1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## 六、栈 / 哈希 / 高频设计

### 20. 最小栈
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
        
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
        
    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
        
    def top(self):
        return self.stack[-1]
        
    def getMin(self):
        return self.min_stack[-1]
```

### 21. LRU 缓存（后端/AI 岗必考）
```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

---

## 七、补全高频题

### 22. 三数之和
```python
def three_sum(nums):
    nums.sort()  # 排序去重核心
    res = []
    for i in range(len(nums)):
        if nums[i] > 0: break  # 正数不可能凑0
        if i > 0 and nums[i] == nums[i-1]: continue  # 去重
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                # 跳过重复值
                while l < r and nums[l] == nums[l+1]: l +=1
                while l < r and nums[r] == nums[r-1]: r -=1
                l +=1
                r -=1
    return res
```

### 23. 移动零
```python
def move_zeroes(nums):
    # 双指针：j 记录非零位置
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[j] = nums[i]
            if i != j:
                nums[i] = 0
            j +=1
```

### 24. 盛最多水的容器
```python
def max_area(height):
    l, r = 0, len(height)-1
    res = 0
    while l < r:
        # 面积 = 宽度 * 较矮的边
        area = (r - l) * min(height[l], height[r])
        res = max(res, area)
        # 移动较短的边（才有可能变大）
        if height[l] < height[r]:
            l +=1
        else:
            r -=1
    return res
```

### 25. 子集
```python
def subsets(nums):
    res = []
    def backtrack(path, start):
        res.append(path.copy())  # 收集所有路径
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(path, i+1)
            path.pop()  # 回溯
    backtrack([], 0)
    return res
```

### 26. 全排列
```python
def permute(nums):
    res = []
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path.copy())
            return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False]*len(nums))
    return res
```

### 27. 二叉树的直径
```python
def diameter_of_binary_tree(root):
    self.res = 0
    def dfs(node):
        if not node: return 0
        left = dfs(node.left)
        right = dfs(node.right)
        self.res = max(self.res, left + right)  # 直径 = 左深度+右深度
        return 1 + max(left, right)  # 返回当前节点深度
    dfs(root)
    return self.res
```

### 28. 路径总和
```python
def has_path_sum(root, target_sum):
    if not root: return False
    # 叶子节点 + 值匹配
    if not root.left and not root.right:
        return root.val == target_sum
    # 递归检查左右子树
    return has_path_sum(root.left, target_sum - root.val) or has_path_sum(root.right, target_sum - root.val)
```

### 29. 只出现一次的数字
```python
def single_number(nums):
    res = 0
    for num in nums:
        res ^= num  # 异或：相同为0，不同为自身，0^n=n
    return res
```

### 30. 多数元素
```python
def majority_element(nums):
    # 摩尔投票法（最优）：时间O(n)，空间O(1)
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    return candidate
```

---

## 面试技巧
1. **先说思路**：写代码前先口头说明核心思路（如"这道题用双指针，左指针...右指针..."）
2. **代码简洁**：优先写核心逻辑，不要过度优化
3. **边界处理**：注意空值、单元素等特殊情况
4. **命名规范**：使用有意义的变量名

这些题目覆盖了 AI 应用开发岗 95% 的手撕场景，熟练掌握后足以应对绝大多数面试。
