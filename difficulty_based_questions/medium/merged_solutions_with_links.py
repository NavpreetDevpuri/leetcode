# Link: https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/description/
class Solution:
  def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
    # Buidl the string in reverse order to avoid creating new copy
    def dfs(root: Optional[TreeNode], val: int, path: List[chr]) -> bool:
      if root.val == val:
        return True
      if root.left and dfs(root.left, val, path):
        path.append('L')
      elif root.right and dfs(root.right, val, path):
        path.append('R')
      return len(path) > 0

    pathToStart = []
    pathToDest = []

    dfs(root, startValue, pathToStart)
    dfs(root, destValue, pathToDest)

    while pathToStart and pathToDest and pathToStart[-1] == pathToDest[-1]:
      pathToStart.pop()
      pathToDest.pop()

    return 'U' * len(pathToStart) + ''.join(reversed(pathToDest))


# Link: https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/description/
class Solution:
  def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
    def lca(root: Optional[TreeNode]) -> Optional[TreeNode]:
      if not root or root.val in (startValue, destValue):
        return root
      left = lca(root.left)
      right = lca(root.right)
      if left and right:
        return root
      return left or right

    def dfs(root: Optional[TreeNode], path: List[chr]) -> None:
      if not root:
        return
      if root.val == startValue:
        self.pathToStart = ''.join(path)
      if root.val == destValue:
        self.pathToDest = ''.join(path)
      path.append('L')
      dfs(root.left, path)
      path.pop()
      path.append('R')
      dfs(root.right, path)
      path.pop()

    dfs(lca(root), [])  # Only this subtree matters.
    return 'U' * len(self.pathToStart) + ''.join(self.pathToDest)


# Link: https://leetcode.com/problems/stock-price-fluctuation/description/
from sortedcontainers import SortedDict


class StockPrice:
  def __init__(self):
    self.timestampToPrice = SortedDict()
    self.pricesCount = SortedDict()

  def update(self, timestamp: int, price: int) -> None:
    if timestamp in self.timestampToPrice:
      prevPrice = self.timestampToPrice[timestamp]
      self.pricesCount[prevPrice] -= 1
      if self.pricesCount[prevPrice] == 0:
        del self.pricesCount[prevPrice]
    self.timestampToPrice[timestamp] = price
    self.pricesCount[price] = self.pricesCount.get(price, 0) + 1

  def current(self) -> int:
    return self.timestampToPrice.peekitem(-1)[1]

  def maximum(self) -> int:
    return self.pricesCount.peekitem(-1)[0]

  def minimum(self) -> int:
    return self.pricesCount.peekitem(0)[0]


# Link: https://leetcode.com/problems/path-sum-ii/description/
class Solution:
  def pathSum(self, root: TreeNode, summ: int) -> List[List[int]]:
    ans = []

    def dfs(root: TreeNode, summ: int, path: List[int]) -> None:
      if not root:
        return
      if root.val == summ and not root.left and not root.right:
        ans.append(path + [root.val])
        return

      dfs(root.left, summ - root.val, path + [root.val])
      dfs(root.right, summ - root.val, path + [root.val])

    dfs(root, summ, [])
    return ans


# Link: https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/description/
class Solution:
  def mctFromLeafValues(self, arr: List[int]) -> int:
    ans = 0
    stack = [math.inf]

    for a in arr:
      while stack and stack[-1] <= a:
        mid = stack.pop()
        # Multiply mid with next greater element in the array,
        # On the left (stack[-1]) or on the right (current number a)
        ans += min(stack[-1], a) * mid
      stack.append(a)

    return ans + sum(a * b for a, b in zip(stack[1:], stack[2:]))


# Link: https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/description/
class Solution:
  def mctFromLeafValues(self, arr: List[int]) -> int:
    ans = 0

    while len(arr) > 1:
      i = arr.index(min(arr))
      ans += min(arr[i - 1:i] + arr[i + 1:i + 2]) * arr.pop(i)

    return ans


# Link: https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/description/
class Solution:
  def mctFromLeafValues(self, arr: List[int]) -> int:
    n = len(arr)
    # dp[i][j] := the minimum cost of arr[i..j]
    dp = [[0] * n for _ in range(n)]
    # maxVal[i][j] := the maximum value of arr[i..j]
    maxVal = [[0] * n for _ in range(n)]

    for i in range(n):
      maxVal[i][i] = arr[i]

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        maxVal[i][j] = max(maxVal[i][j - 1], maxVal[i + 1][j])

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        dp[i][j] = math.inf
        for k in range(i, j):
          dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] +
                         maxVal[i][k] * maxVal[k + 1][j])

    return dp[0][-1]


# Link: https://leetcode.com/problems/determine-if-two-strings-are-close/description/
class Solution:
  def closeStrings(self, word1: str, word2: str) -> bool:
    if len(word1) != len(word2):
      return False

    count1 = collections.Counter(word1)
    count2 = collections.Counter(word2)
    if count1.keys() != count2.keys():
      return False

    return sorted(count1.values()) == sorted(count2.values())


# Link: https://leetcode.com/problems/alice-and-bob-playing-flower-game/description/
class Solution:
  def flowerGame(self, n: int, m: int) -> int:
    # Alice wins if x + y is odd, occurring when:
    #   1. x is even and y is odd, or
    #   2. y is even and x is odd.
    xEven = n // 2
    yEven = m // 2
    xOdd = (n + 1) // 2
    yOdd = (m + 1) // 2
    return xEven * yOdd + yEven * xOdd


# Link: https://leetcode.com/problems/number-of-times-binary-string-is-prefix-aligned/description/
class Solution:
  def numTimesAllBlue(self, flips: List[int]) -> int:
    ans = 0
    rightmost = 0

    for i, flip in enumerate(flips):
      rightmost = max(rightmost, flip)
      # max(flips[0..i]) = rightmost = i + 1,
      # so flips[0..i] is a permutation of 1, 2, ..., i + 1.
      if rightmost == i + 1:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/palindrome-partitioning/description/
class Solution:
  def partition(self, s: str) -> List[List[str]]:
    ans = []

    def isPalindrome(s: str) -> bool:
      return s == s[::-1]

    def dfs(s: str, j: int, path: List[str], ans: List[List[str]]) -> None:
      if j == len(s):
        ans.append(path)
        return

      for i in range(j, len(s)):
        if isPalindrome(s[j: i + 1]):
          dfs(s, i + 1, path + [s[j: i + 1]], ans)

    dfs(s, 0, [], ans)
    return ans


# Link: https://leetcode.com/problems/minimum-number-of-coins-to-be-added/description/
class Solution:
  # Same as 330. Patching Array
  def minimumAddedCoins(self, coins: List[int], target: int) -> int:
    ans = 0
    i = 0  # coins' index
    miss = 1  # the minimum sum in [1, n] we might miss

    coins.sort()

    while miss <= target:
      if i < len(coins) and coins[i] <= miss:
        miss += coins[i]
        i += 1
      else:
        # Greedily add `miss` itself to increase the range from
        # [1, miss) to [1, 2 * miss).
        miss += miss
        ans += 1

    return ans


# Link: https://leetcode.com/problems/finding-the-users-active-minutes/description/
class Solution:
  def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
    idToTimes = collections.defaultdict(set)

    for id, time in logs:
      idToTimes[id].add(time)

    c = collections.Counter(len(times) for times in idToTimes.values())
    return [c[i] for i in range(1, k + 1)]


# Link: https://leetcode.com/problems/next-greater-numerically-balanced-number/description/
class Solution:
  def nextBeautifulNumber(self, n: int) -> int:
    def isBalance(num: int) -> bool:
      count = [0] * 10
      while num:
        if num % 10 == 0:
          return False
        count[num % 10] += 1
        num //= 10
      return all(c == i for i, c in enumerate(count) if c)

    n += 1
    while not isBalance(n):
      n += 1
    return n


# Link: https://leetcode.com/problems/maximum-binary-string-after-change/description/
class Solution:
  def maximumBinaryString(self, binary: str) -> str:
    #     e.g. binary = '100110'
    # Do Operation 2 -> '100011'
    # Do Operation 1 -> '111011'
    # So, the index of the only '0' is prefixOnes + zeros - 1.
    zeros = binary.count('0')
    prefixOnes = binary.find('0')

    # Make the entire string as 1s.
    ans = ['1'] * len(binary)

    # Make the only '0' if necessary.
    if prefixOnes != -1:
      ans[prefixOnes + zeros - 1] = '0'
    return ''.join(ans)


# Link: https://leetcode.com/problems/largest-divisible-subset/description/
class Solution:
  def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = []
    count = [1] * n
    prevIndex = [-1] * n
    maxCount = 0
    index = -1

    nums.sort()

    for i, num in enumerate(nums):
      for j in reversed(range(i)):
        if num % nums[j] == 0 and count[i] < count[j] + 1:
          count[i] = count[j] + 1
          prevIndex[i] = j
      if count[i] > maxCount:
        maxCount = count[i]
        index = i

    while index != -1:
      ans.append(nums[index])
      index = prevIndex[index]

    return ans


# Link: https://leetcode.com/problems/rotate-function/description/
class Solution:
  def maxRotateFunction(self, nums: List[int]) -> int:
    f = sum(i * num for i, num in enumerate(nums))
    ans = f
    summ = sum(nums)

    for a in reversed(nums):
      f += summ - len(nums) * a
      ans = max(ans, f)

    return ans


# Link: https://leetcode.com/problems/design-sql/description/
class SQL:
  def __init__(self, names: List[str], columns: List[int]):
    self.db: Dict[str, List[List[str]]] = collections.defaultdict(list)

  def insertRow(self, name: str, row: List[str]) -> None:
    self.db[name].append(row)

  def deleteRow(self, name: str, rowId: int) -> None:
    pass

  def selectCell(self, name: str, rowId: int, columnId: int) -> str:
    return self.db[name][rowId - 1][columnId - 1]


# Link: https://leetcode.com/problems/linked-list-components/description/
class Solution:
  def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
    ans = 0
    numsSet = set(nums)

    while head:
      if head.val in numsSet and (head.next == None or head.next.val not in numsSet):
        ans += 1
      head = head.next

    return ans


# Link: https://leetcode.com/problems/special-permutations/description/
class Solution:
  def specialPerm(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    maxMask = 1 << len(nums)

    @functools.lru_cache(None)
    def dp(prev: int, mask: int) -> int:
      """
      Returns the number of special permutations, where the previous number is
      nums[i] and `mask` is the bitmask of the used numbers.
      """
      if mask == maxMask - 1:
        return 1

      res = 0

      for i, num in enumerate(nums):
        if mask >> i & 1:
          continue
        if num % nums[prev] == 0 or nums[prev] % num == 0:
          res += dp(i, mask | 1 << i)
          res %= kMod

      return res

    return sum(dp(i, 1 << i)
               for i in range(len(nums))) % kMod


# Link: https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/
class Solution:
  def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    ans = -1
    minCitiesCount = n
    dist = self._floydWarshall(n, edges, distanceThreshold)

    for i in range(n):
      citiesCount = sum(dist[i][j] <= distanceThreshold for j in range(n))
      if citiesCount <= minCitiesCount:
        ans = i
        minCitiesCount = citiesCount

    return ans

  def _floydWarshall(self, n: int, edges: List[List[int]], distanceThreshold: int) -> List[List[int]]:
    dist = [[distanceThreshold + 1] * n for _ in range(n)]

    for i in range(n):
      dist[i][i] = 0

    for u, v, w in edges:
      dist[u][v] = w
      dist[v][u] = w

    for k in range(n):
      for i in range(n):
        for j in range(n):
          dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


# Link: https://leetcode.com/problems/longest-word-with-all-prefixes/description/
class Solution:
  def __init__(self):
    self.root = {}

  def longestWord(self, words: List[str]) -> str:
    ans = ''

    for word in words:
      self.insert(word)

    for word in words:
      if not self.allPrefixed(word):
        continue
      if len(ans) < len(word) or (len(ans) == len(word) and ans > word):
        ans = word

    return ans

  def insert(self, word: str) -> None:
    node = self.root
    for c in word:
      if c not in node:
        node[c] = {}
      node = node[c]
    node['isWord'] = True

  def allPrefixed(self, word: str) -> bool:
    node = self.root
    for c in word:
      node = node[c]
      if 'isWord' not in node:
        return False
    return True


# Link: https://leetcode.com/problems/beautiful-towers-i/description/
class Solution:
  def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
    n = len(maxHeights)
    maxSum = [0] * n  # maxSum[i] := the maximum sum with peak i

    def process(stack: List[int], i: int, summ: int) -> int:
      while len(stack) > 1 and maxHeights[stack[-1]] > maxHeights[i]:
        j = stack.pop()
        # The last abs(j - stack[-1]) heights are maxHeights[j].
        summ -= abs(j - stack[-1]) * maxHeights[j]
      # Put abs(i - stack[-1]) `maxHeight` in heights.
      summ += abs(i - stack[-1]) * maxHeights[i]
      stack.append(i)
      return summ

    stack = [-1]
    summ = 0
    for i in range(len(maxHeights)):
      summ = process(stack, i, summ)
      maxSum[i] = summ

    stack = [n]
    summ = 0
    for i in range(n - 1, -1, -1):
      summ = process(stack, i, summ)
      maxSum[i] += summ - maxHeights[i]

    return max(maxSum)


# Link: https://leetcode.com/problems/find-the-original-array-of-prefix-xor/description/
class Solution:
  def findArray(self, pref: List[int]) -> List[int]:
    ans = [0] * len(pref)

    ans[0] = pref[0]
    for i in range(1, len(ans)):
      ans[i] = pref[i] ^ pref[i - 1]

    return ans


# Link: https://leetcode.com/problems/minimum-additions-to-make-valid-string/description/
class Solution:
  def addMinimum(self, word: str) -> int:
    letters = ['a', 'b', 'c']
    ans = 0
    i = 0

    while i < len(word):
      for c in letters:
        if i < len(word) and word[i] == c:
          i += 1
        else:
          ans += 1

    return ans


# Link: https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/
class Solution:
  def maxLevelSum(self, root: Optional[TreeNode]) -> int:
    ans = 1
    maxLevelSum = -math.inf
    q = collections.deque([root])

    level = 1
    while q:
      levelSum = 0
      for _ in range(len(q)):
        node = q.popleft()
        levelSum += node.val
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      if maxLevelSum < levelSum:
        maxLevelSum = levelSum
        ans = level
      level += 1

    return ans


# Link: https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/
class Solution:
  def maxLevelSum(self, root: Optional[TreeNode]) -> int:
    # levelSums[i] := the sum of level (i + 1) (1-indexed)
    levelSums = []

    def dfs(root: Optional[TreeNode], level: int) -> None:
      if not root:
        return
      if len(levelSums) == level:
        levelSums.append(0)
      levelSums[level] += root.val
      dfs(root.left, level + 1)
      dfs(root.right, level + 1)

    dfs(root, 0)
    return 1 + levelSums.index(max(levelSums))


# Link: https://leetcode.com/problems/3sum/description/
class Solution:
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    if len(nums) < 3:
      return []

    ans = []

    nums.sort()

    for i in range(len(nums) - 2):
      if i > 0 and nums[i] == nums[i - 1]:
        continue
      # Choose nums[i] as the first number in the triplet, then search the
      # remaining numbers in [i + 1, n - 1].
      l = i + 1
      r = len(nums) - 1
      while l < r:
        summ = nums[i] + nums[l] + nums[r]
        if summ == 0:
          ans.append((nums[i], nums[l], nums[r]))
          l += 1
          r -= 1
          while nums[l] == nums[l - 1] and l < r:
            l += 1
          while nums[r] == nums[r + 1] and l < r:
            r -= 1
        elif summ < 0:
          l += 1
        else:
          r -= 1

    return ans


# Link: https://leetcode.com/problems/lonely-pixel-i/description/
class Solution:
  def findLonelyPixel(self, picture: List[List[str]]) -> int:
    m = len(picture)
    n = len(picture[0])
    ans = 0
    rows = [0] * m  # rows[i] := the number of B's in rows i
    cols = [0] * n  # cols[i] := the number of B's in cols i

    for i in range(m):
      for j in range(n):
        if picture[i][j] == 'B':
          rows[i] += 1
          cols[j] += 1

    for i in range(m):
      if rows[i] == 1:  # Only have to examine the rows if rows[i] == 1.
        for j in range(n):
          # After meeting a 'B' in this rows, break and search the next row.
          if picture[i][j] == 'B':
            if cols[j] == 1:
              ans += 1
            break

    return ans


# Link: https://leetcode.com/problems/maximum-size-of-a-set-after-removals/description/
class Solution:
  def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
    set1 = set(nums1)
    set2 = set(nums2)
    common = set1.intersection(set2)

    n = len(nums1)
    n1 = len(set1)
    n2 = len(set2)
    nc = len(common)
    maxUniqueNums1 = min(n1 - nc, n // 2)
    maxUniqueNums2 = min(n2 - nc, n // 2)
    return min(n, maxUniqueNums1 + maxUniqueNums2 + nc)


# Link: https://leetcode.com/problems/shortest-string-that-contains-three-strings/description/
class Solution:
  def minimumString(self, a: str, b: str, c: str) -> str:
    def merge(a: str, b: str) -> str:
      """Merges a and b."""
      if a in b:  # a is a substring of b.
        return b
      for i in range(len(a)):
        aSuffix = a[i:]
        bPrefix = b[:len(aSuffix)]
        if aSuffix == bPrefix:
          return a + b[len(bPrefix):]
      return a + b

    abc = merge(a, merge(b, c))
    acb = merge(a, merge(c, b))
    bac = merge(b, merge(a, c))
    bca = merge(b, merge(c, a))
    cab = merge(c, merge(a, b))
    cba = merge(c, merge(b, a))
    return self._getMin([abc, acb, bac, bca, cab, cba])

  def _getMin(self, words: List[str]) -> str:
    """Returns the lexicographically smallest string."""

    def getMin(a: str, b: str) -> str:
      """Returns the lexicographically smaller string."""
      return a if len(a) < len(b) or (len(a) == len(b) and a < b) else b

    res = words[0]
    for i in range(1, len(words)):
      res = getMin(res, words[i])
    return res


# Link: https://leetcode.com/problems/most-profit-assigning-work/description/
class Solution:
  def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
    ans = 0
    jobs = sorted(zip(difficulty, profit))
    worker.sort(reverse=1)

    i = 0
    maxProfit = 0

    for w in sorted(worker):
      while i < len(jobs) and w >= jobs[i][0]:
        maxProfit = max(maxProfit, jobs[i][1])
        i += 1
      ans += maxProfit

    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-make-all-array-elements-equal/description/
class Solution:
  def minOperations(self, nums: List[int], queries: List[int]) -> List[int]:
    n = len(nums)
    nums.sort()
    prefix = [0] + list(itertools.accumulate(nums))
    splits = [(query, bisect.bisect_right(nums, query)) for query in queries]
    return [(query * i - prefix[i]) +
            (prefix[-1] - prefix[i] - query * (n - i))
            for query, i in splits]


# Link: https://leetcode.com/problems/invalid-transactions/description/
class Solution:
  def invalidTransactions(self, transactions: List[str]) -> List[str]:
    ans = []
    nameToTranses = collections.defaultdict(list)

    for t in transactions:
      name, time, amount, city = t.split(',')
      time, amount = int(time), int(amount)
      nameToTranses[name].append({'time': time, 'city': city})

    for t in transactions:
      name, time, amount, city = t.split(',')
      time, amount = int(time), int(amount)
      if amount > 1000:
        ans.append(t)
      elif name in nameToTranses:
        for sameName in nameToTranses[name]:
          if abs(sameName['time'] - time) <= 60 and sameName['city'] != city:
            ans.append(t)
            break

    return ans


# Link: https://leetcode.com/problems/minimum-subarrays-in-a-valid-split/description/
class Solution:
  def validSubarraySplit(self, nums: List[int]) -> int:
    # dp[i] := the minimum number of subarrays to validly split nums[0..i]
    dp = [math.inf] * len(nums)

    for i, num in enumerate(nums):
      for j in range(i + 1):
        if math.gcd(nums[j], num) > 1:
          dp[i] = min(dp[i], 1 if j == 0 else dp[j - 1] + 1)

    return -1 if dp[-1] == math.inf else dp[-1]


# Link: https://leetcode.com/problems/count-positions-on-street-with-required-brightness/description/
class Solution:
  def meetRequirement(self, n: int, lights: List[List[int]], requirement: List[int]) -> int:
    ans = 0
    currBrightness = 0
    change = [0] * (n + 1)

    for position, rg in lights:
      change[max(0, position - rg)] += 1
      change[min(n, position + rg + 1)] -= 1

    for i in range(n):
      currBrightness += change[i]
      if currBrightness >= requirement[i]:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/maximum-matrix-sum/description/
class Solution:
  def maxMatrixSum(self, matrix: List[List[int]]) -> int:
    absSum = 0
    minAbs = math.inf
    # 0 := even number of negatives
    # 1 := odd number of negatives
    oddNeg = 0

    for row in matrix:
      for num in row:
        absSum += abs(num)
        minAbs = min(minAbs, abs(num))
        if num < 0:
          oddNeg ^= 1

    return absSum - oddNeg * minAbs * 2


# Link: https://leetcode.com/problems/grid-game/description/
class Solution:
  def gridGame(self, grid: List[List[int]]) -> int:
    n = len(grid[0])
    ans = math.inf
    sumRow0 = sum(grid[0])
    sumRow1 = 0

    for i in range(n):
      sumRow0 -= grid[0][i]
      ans = min(ans, max(sumRow0, sumRow1))
      sumRow1 += grid[1][i]

    return ans


# Link: https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/description/
class Solution:
  def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
    root = TreeNode(preorder[0])
    stack = [root]

    for i in range(1, len(preorder)):
      parent = stack[-1]
      child = TreeNode(preorder[i])
      # Adjust the parent.
      while stack and stack[-1].val < child.val:
        parent = stack.pop()
      # Create parent-child link according to BST property.
      if parent.val > child.val:
        parent.left = child
      else:
        parent.right = child
      stack.append(child)

    return root


# Link: https://leetcode.com/problems/find-palindrome-with-fixed-length/description/
class Solution:
  def kthPalindrome(self, queries: List[int], intLength: int) -> List[int]:
    start = pow(10, (intLength + 1) // 2 - 1)
    end = pow(10, (intLength + 1) // 2)
    mul = pow(10, intLength // 2)

    def reverse(num: int) -> int:
      res = 0
      while num:
        res = res * 10 + num % 10
        num //= 10
      return res

    def getKthPalindrome(query: int) -> int:
      prefix = start + query - 1
      return prefix * mul + reverse(prefix // 10 if intLength & 1 else prefix)

    return [-1 if start + query > end else getKthPalindrome(query)
            for query in queries]


# Link: https://leetcode.com/problems/minimum-deletions-to-make-array-beautiful/description/
class Solution:
  def minDeletion(self, nums: List[int]) -> int:
    ans = 0

    for i in range(len(nums) - 1):
      # i - ans := the index after deletion
      if nums[i] == nums[i + 1] and (i - ans) % 2 == 0:
        ans += 1

    # Add one if the length after deletion is odd
    return ans + ((len(nums) - ans) & 1)


# Link: https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-ii/description/
class Solution:
  # Same as 2873. Maximum Value of an Ordered Triplet I
  def maximumTripletValue(self, nums: List[int]) -> int:
    ans = 0
    maxDiff = 0  # max(nums[i] - nums[j])
    maxNum = 0   # max(nums[i])

    for num in nums:
      ans = max(ans, maxDiff * num)         # num := nums[k]
      maxDiff = max(maxDiff, maxNum - num)  # num := nums[j]
      maxNum = max(maxNum, num)             # num := nums[i]

    return ans


# Link: https://leetcode.com/problems/determine-if-a-cell-is-reachable-at-a-given-time/description/
class Solution:
  def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
    minStep = max(abs(sx - fx), abs(sy - fy))
    return t != 1 if minStep == 0 else minStep <= t


# Link: https://leetcode.com/problems/double-modular-exponentiation/description/
class Solution:
  def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
    return [i for i, (a, b, c, m) in enumerate(variables)
            if pow(pow(a, b, 10), c, m) == target]


# Link: https://leetcode.com/problems/target-sum/description/
class Solution:
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    summ = sum(nums)
    if summ < abs(target) or (summ + target) & 1:
      return 0

    def knapsack(target: int) -> int:
      # dp[i] := the number of ways to sum to i by nums so far
      dp = [1] + [0] * summ

      for num in nums:
        for j in range(summ, num - 1, -1):
          dp[j] += dp[j - num]

      return dp[target]

    return knapsack((summ + target) // 2)


# Link: https://leetcode.com/problems/target-sum/description/
class Solution:
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    summ = sum(nums)
    if summ < abs(target) or (summ + target) & 1:
      return 0

    def knapsack(nums: List[int], target: int) -> int:
      # dp[i] := the number of ways to sum to i by nums so far
      dp = [0] * (target + 1)
      dp[0] = 1

      for num in nums:
        for i in range(target, num - 1, -1):
          dp[i] += dp[i - num]

      return dp[target]

    return knapsack(nums, (summ + target) // 2)


# Link: https://leetcode.com/problems/construct-smallest-number-from-di-string/description/
class Solution:
  def smallestNumber(self, pattern: str) -> str:
    ans = []
    stack = ['1']

    for c in pattern:
      maxSorFar = stack[-1]
      if c == 'I':
        while stack:
          maxSorFar = max(maxSorFar, stack[-1])
          ans.append(stack.pop())
      stack.append(chr(ord(maxSorFar) + 1))

    while stack:
      ans.append(stack.pop())

    return ''.join(ans)


# Link: https://leetcode.com/problems/construct-k-palindrome-strings/description/
class Solution:
  def canConstruct(self, s: str, k: int) -> bool:
    # If |s| < k, we cannot construct k strings from the s.
    # If the number of letters that have odd counts > k, the minimum number of
    # palindromic strings we can construct is > k.
    return sum(freq & 1
               for freq in collections.Counter(s).values()) <= k <= len(s)


# Link: https://leetcode.com/problems/find-the-grid-of-region-average/description/
class Solution:
  def resultGrid(self, image: List[List[int]], threshold: int) -> List[List[int]]:
    m = len(image)
    n = len(image[0])
    sums = [[0] * n for _ in range(m)]
    counts = [[0] * n for _ in range(m)]

    for i in range(m - 2):
      for j in range(n - 2):
        if self._isRegion(image, i, j, threshold):
          subgridSum = sum(image[x][y]
                           for x in range(i, i + 3)
                           for y in range(j, j + 3))
          for x in range(i, i + 3):
            for y in range(j, j + 3):
              sums[x][y] += subgridSum // 9
              counts[x][y] += 1

    for i in range(m):
      for j in range(n):
        if counts[i][j] > 0:
          image[i][j] = sums[i][j] // counts[i][j]

    return image

  def _isRegion(self, image: List[List[int]], i: int, j: int, threshold: int) -> bool:
    """Returns True if image[i..i + 2][j..j + 2] is a region."""
    for x in range(i, i + 3):
      for y in range(j, j + 3):
        if x > i and abs(image[x][y] - image[x - 1][y]) > threshold:
          return False
        if y > j and abs(image[x][y] - image[x][y - 1]) > threshold:
          return False
    return True


# Link: https://leetcode.com/problems/sum-root-to-leaf-numbers/description/
class Solution:
  def sumNumbers(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def dfs(root: Optional[TreeNode], path: int) -> None:
      nonlocal ans
      if not root:
        return
      if not root.left and not root.right:
        ans += path * 10 + root.val
        return

      dfs(root.left, path * 10 + root.val)
      dfs(root.right, path * 10 + root.val)

    dfs(root, 0)
    return ans


# Link: https://leetcode.com/problems/pour-water-between-buckets-to-make-water-levels-equal/description/
class Solution:
  def equalizeWater(self, buckets: List[int], loss: int) -> float:
    kErr = 1e-5
    kPercentage = (100 - loss) / 100
    l = 0.0
    r = max(buckets)

    def canFill(target: float) -> bool:
      extra = 0
      need = 0
      for bucket in buckets:
        if bucket > target:
          extra += bucket - target
        else:
          need += target - bucket
      return extra * kPercentage >= need

    while r - l > kErr:
      m = (l + r) / 2
      if canFill(m):
        l = m
      else:
        r = m

    return l


# Link: https://leetcode.com/problems/find-the-number-of-ways-to-place-people-i/description/
class Solution:
  def numberOfPairs(self, points: List[List[int]]) -> int:
    ans = 0

    points.sort(key=lambda x: (x[0], -x[1]))

    for i, (_, yi) in enumerate(points):
      maxY = -math.inf
      for j in range(i + 1, len(points)):
        _, yj = points[j]
        # Chisato is in the upper-left corner at (xi, yi), and Takina is in the
        # lower-right corner at (xj, yj). Also, if yj > maxY, it means that
        # nobody other than Chisato and Takina is inside or on the fence.
        if yi >= yj > maxY:
          ans += 1
          maxY = yj

    return ans


# Link: https://leetcode.com/problems/construct-string-with-repeat-limit/description/
class Solution:
  def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
    ans = ''
    count = collections.Counter(s)

    while True:
      addOne = ans and self._shouldAddOne(ans, count)
      c = self._getLargestChar(ans, count)
      if c == ' ':
        break
      repeats = 1 if addOne else min(count[c], repeatLimit)
      ans += c * repeats
      count[c] -= repeats

    return ans

  def _shouldAddOne(self, ans: str, count: collections.Counter) -> bool:
    for c in reversed(string.ascii_lowercase):
      if count[c]:
        return ans[-1] == c
    return False

  def _getLargestChar(self, ans: str, count: collections.Counter) -> int:
    for c in reversed(string.ascii_lowercase):
      if count[c] and (not ans or ans[-1] != c):
        return c
    return ' '


# Link: https://leetcode.com/problems/number-of-strings-which-can-be-rearranged-to-contain-substring/description/
class Solution:
  def stringCount(self, n: int) -> int:
    # There're three invalid conditions:
    #   a. count('l') == 0
    #   b. count('e') < 2
    #   c. count('t') == 0
    #
    # By Principle of Inclusion-Exclusion (PIE):
    #   ans = allCount - a - b - c + ab + ac + bc - abc
    kMod = 1_000_000_007
    allCount = pow(26, n, kMod)
    a = pow(25, n, kMod)
    b = pow(25, n, kMod)
    c = pow(25, n, kMod) + n * pow(25, n - 1, kMod)
    ab = pow(24, n, kMod) + n * pow(24, n - 1, kMod)
    ac = pow(24, n, kMod)
    bc = pow(24, n, kMod) + n * pow(24, n - 1, kMod)
    abc = pow(23, n, kMod) + n * pow(23, n - 1, kMod)
    return (allCount - a - b - c + ab + ac + bc - abc) % kMod


# Link: https://leetcode.com/problems/count-substrings-without-repeating-character/description/
class Solution:
  def numberOfSpecialSubstrings(self, s: str) -> int:
    ans = 0
    count = collections.Counter()

    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      while count[c] == 2:
        count[s[l]] -= 1
        l += 1
      ans += r - l + 1

    return ans


# Link: https://leetcode.com/problems/reach-a-number/description/
class Solution:
  def reachNumber(self, target: int) -> int:
    ans = 0
    pos = 0
    target = abs(target)

    while pos < target:
      ans += 1
      pos += ans

    while (pos - target) & 1:
      ans += 1
      pos += ans

    return ans


# Link: https://leetcode.com/problems/design-an-atm-machine/description/
class ATM:
  def __init__(self):
    self.banknotes = [20, 50, 100, 200, 500]
    self.bank = [0] * 5

  def deposit(self, banknotesCount: List[int]) -> None:
    for i in range(5):
      self.bank[i] += banknotesCount[i]

  def withdraw(self, amount: int) -> List[int]:
    withdrew = [0] * 5

    for i in reversed(range(5)):
      withdrew[i] = min(self.bank[i], amount // self.banknotes[i])
      amount -= withdrew[i] * self.banknotes[i]

    if amount:
      return [-1]

    for i in range(5):
      self.bank[i] -= withdrew[i]
    return withdrew


# Link: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
class Solution:
  def sortedListToBST(self, head: ListNode) -> TreeNode:
    def findMid(head: ListNode) -> ListNode:
      prev = None
      slow = head
      fast = head

      while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
      prev.next = None

      return slow

    if not head:
      return None
    if not head.next:
      return TreeNode(head.val)

    mid = findMid(head)
    root = TreeNode(mid.val)
    root.left = self.sortedListToBST(head)
    root.right = self.sortedListToBST(mid.next)

    return root


# Link: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
class Solution:
  def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
    A = []

    # Construct the array.
    curr = head
    while curr:
      A.append(curr.val)
      curr = curr.next

    def helper(l: int, r: int) -> Optional[TreeNode]:
      if l > r:
        return None

      m = (l + r) // 2
      root = TreeNode(A[m])
      root.left = helper(l, m - 1)
      root.right = helper(m + 1, r)
      return root

    return helper(0, len(A) - 1)


# Link: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
class Solution:
  def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
    def helper(l: int, r: int) -> Optional[TreeNode]:
      nonlocal head
      if l > r:
        return None

      m = (l + r) // 2

      # Simulate inorder traversal: recursively form the left half.
      left = helper(l, m - 1)

      # Once the left half is traversed, process the current node.
      root = TreeNode(head.val)
      root.left = left

      # Maintain the invariance.
      head = head.next

      # Simulate inorder traversal: recursively form the right half.
      root.right = helper(m + 1, r)
      return root

    return helper(0, self._getLength(head) - 1)

  def _getLength(self, head: Optional[ListNode]) -> int:
    length = 0
    curr = head
    while curr:
      length += 1
      curr = curr.next
    return length


# Link: https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
class Solution:
  def lengthOfLongestSubstring(self, s: str) -> int:
    ans = 0
    count = collections.Counter()

    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      while count[c] > 1:
        count[s[l]] -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
class Solution:
  def lengthOfLongestSubstring(self, s: str) -> int:
    ans = 0
    # The substring s[j + 1..i] has no repeating characters.
    j = -1
    # lastSeen[c] := the index of the last time c appeared
    lastSeen = {}

    for i, c in enumerate(s):
      # Update j to lastSeen[c], so the window must start from j + 1.
      j = max(j, lastSeen.get(c, -1))
      ans = max(ans, i - j)
      lastSeen[c] = i

    return ans


# Link: https://leetcode.com/problems/median-of-a-row-wise-sorted-matrix/description/
class Solution:
  def matrixMedian(self, grid: List[List[int]]) -> int:
    noGreaterThanMedianCount = len(grid) * len(grid[0]) // 2 + 1
    l = 1
    r = int(1e6)

    while l < r:
      m = (l + r) // 2
      if sum(bisect_right(row, m) for row in grid) >= \
              noGreaterThanMedianCount:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/description/
class Solution:
  def longestSubsequence(self, s: str, k: int) -> int:
    oneCount = 0
    num = 0
    pow = 1

    # Take as many 1s as possible from the right.
    for i in reversed(range(len(s))):
      if num + pow > k:
        break
      if s[i] == '1':
        oneCount += 1
        num += pow
      pow *= 2

    return s.count('0') + oneCount


# Link: https://leetcode.com/problems/escape-the-ghosts/description/
class Solution:
  def escapeGhosts(self, ghosts: List[List[int]], target: List[int]) -> bool:
    ghostSteps = min(abs(x - target[0]) +
                     abs(y - target[1]) for x, y in ghosts)

    return abs(target[0]) + abs(target[1]) < ghostSteps


# Link: https://leetcode.com/problems/nested-list-weight-sum/description/
class Solution:
  def depthSum(self, nestedList: List[NestedInteger]) -> int:
    ans = 0
    depth = 0
    q = collections.deque()

    def addIntegers(nestedList: List[NestedInteger]) -> None:
      for ni in nestedList:
        q.append(ni)

    addIntegers(nestedList)

    while q:
      depth += 1
      for _ in range(len(q)):
        ni = q.popleft()
        if ni.isInteger():
          ans += ni.getInteger() * depth
        else:
          addIntegers(ni.getList())

    return ans


# Link: https://leetcode.com/problems/nested-list-weight-sum/description/
class Solution:
  def depthSum(self, nestedList: List[NestedInteger]) -> int:
    ans = 0

    def dfs(nestedList: List[NestedInteger], depth: int) -> None:
      nonlocal ans
      for ni in nestedList:
        if ni.isInteger():
          ans += ni.getInteger() * depth
        else:
          dfs(ni.getList(), depth + 1)

    dfs(nestedList, 1)
    return ans


# Link: https://leetcode.com/problems/the-number-of-beautiful-subsets/description/
# e.g. nums = [2, 3, 4, 4], k = 2
#
# subset[0] = [2, 4, 4']
# subset[1] = [1]
# count = {2: 1, 4: 2, 1: 1}
#
# Initially, skip = len([]) = 0, pick = len([]) = 0
#
# * For values in subset[0]:
#   After 2:
#     skip = skip + pick = len([]) = 0
#     pick = (2^count[2] - 1) * (1 + skip + pick)
#          = len([[2]]) * len([[]])
#          = len([[2]]) = 1
#   After 4:
#     skip = skip + pick = len([[2]]) = 1
#     pick = (2^count[4] - 1) * (1 + skip)
#          = len([[4], [4'], [4, 4']]) * len([[]])
#          = len([[4], [4'], [4, 4']]) = 3
#
# * For values in subset[1]:
#   After 1:
#     skip = skip + pick
#          = len([[2], [4], [4'], [4, 4']]) = 4
#     pick = (2^count[1] - 1) * (1 + skip + pick)
#          = len([[1]]) * len([[], [2], [4], [4'], [4, 4']])
#          = len([[1], [1, 2], [1, 4], [1, 4'], [1, 4, 4']]) = 5
#
# So, ans = skip + pick = 9

class Solution:
  def beautifulSubsets(self, nums: List[int], k: int) -> int:
    count = collections.Counter(nums)
    modToSubset = collections.defaultdict(set)

    for num in nums:
      modToSubset[num % k].add(num)

    prevNum = -k
    skip = 0
    pick = 0

    for subset in modToSubset.values():
      for num in sorted(subset):
        nonEmptyCount = 2**count[num] - 1
        skip, pick = skip + pick, nonEmptyCount * \
            (1 + skip + (0 if num - prevNum == k else pick))
        prevNum = num

    return skip + pick


# Link: https://leetcode.com/problems/uncrossed-lines/description/
class Solution:
  def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
    m = len(nums1)
    n = len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        dp[i][j] = dp[i - 1][j - 1] + 1 \
            if nums1[i - 1] == nums2[j - 1] \
            else max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# Link: https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/description/
class Solution:
  # Very similar to 53. Maximum Subarray
  def maximumSum(self, arr: List[int]) -> int:
    ans = -math.inf
    zero = -math.inf  # no deletion
    one = -math.inf   # <= 1 deletion

    for a in arr:
      one = max(a, one + a, zero)
      zero = max(a, zero + a)
      ans = max(ans, one)

    return ans


# Link: https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion/description/
class Solution:
  # Very similar to 53. Maximum Subarray
  def maximumSum(self, arr: List[int]) -> int:
    # dp[0][i] := the maximum sum subarray ending in i (no deletion)
    # dp[1][i] := the maximum sum subarray ending in i (at most 1 deletion)
    dp = [[0] * len(arr) for _ in range(2)]

    dp[0][0] = arr[0]
    dp[1][0] = arr[0]
    for i in range(1, len(arr)):
      dp[0][i] = max(arr[i], dp[0][i - 1] + arr[i])
      dp[1][i] = max(arr[i], dp[1][i - 1] + arr[i], dp[0][i - 1])

    return max(dp[1])


# Link: https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/
class Solution:
  # Similar to 3. Longest SubWithout Repeating Characters
  def numberOfSubstrings(self, s: str) -> int:
    ans = 0
    # lastSeen[c] := the index of the last time c appeared
    lastSeen = {c: -1 for c in 'abc'}

    for i, c in enumerate(s):
      lastSeen[c] = i
      # s[0..i], s[1..i], s[min(lastSeen)..i] are satisfied strings.
      ans += 1 + min(lastSeen.values())

    return ans


# Link: https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/
class Solution:
  # Similar to 3. Longest SubWithout Repeating Characters
  def numberOfSubstrings(self, s: str) -> int:
    ans = 0
    count = {c: 0 for c in 'abc'}

    l = 0
    for c in s:
      count[c] += 1
      while min(count.values()) > 0:
        count[s[l]] -= 1
        l += 1
      # s[0..r], s[1..r], ..., s[l - 1..r] are satified strings.
      ans += l

    return ans


# Link: https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/description/
class Solution:
  def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
    arr.sort()
    arr[0] = 1

    for i in range(1, len(arr)):
      arr[i] = min(arr[i], arr[i - 1] + 1)

    return arr[-1]


# Link: https://leetcode.com/problems/number-of-islands/description/
class Solution:
  def numIslands(self, grid: List[List[str]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])

    def bfs(r, c):
      q = collections.deque([(r, c)])
      grid[r][c] = '2'  # Mark '2' as visited.
      while q:
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if grid[x][y] != '1':
            continue
          q.append((x, y))
          grid[x][y] = '2'  # Mark '2' as visited.

    ans = 0

    for i in range(m):
      for j in range(n):
        if grid[i][j] == '1':
          bfs(i, j)
          ans += 1

    return ans


# Link: https://leetcode.com/problems/number-of-islands/description/
class Solution:
  def numIslands(self, grid: List[List[str]]) -> int:
    m = len(grid)
    n = len(grid[0])

    def dfs(i: int, j: int) -> None:
      if i < 0 or i == m or j < 0 or j == n:
        return
      if grid[i][j] != '1':
        return

      grid[i][j] = '2'  # Mark '2' as visited.
      dfs(i + 1, j)
      dfs(i - 1, j)
      dfs(i, j + 1)
      dfs(i, j - 1)

    ans = 0

    for i in range(m):
      for j in range(n):
        if grid[i][j] == '1':
          dfs(i, j)
          ans += 1

    return ans


# Link: https://leetcode.com/problems/car-fleet/description/
class Solution:
  def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
    ans = 0
    times = [
        float(target - p) / s for p, s in sorted(zip(position, speed),
                                                 reverse=True)]
    maxTime = 0  # the time of the slowest car to reach the target

    for time in times:
      # A car needs more time to reach the target, so it becomes the slowest.
      if time > maxTime:
        maxTime = time
        ans += 1

    return ans


# Link: https://leetcode.com/problems/longest-palindromic-subsequence-ii/description/
class Solution:
  def longestPalindromeSubseq(self, s: str) -> int:
    n = len(s)
    # dp[i][j][k] := the length of LPS(s[i..j]), where the previous letter is
    # ('a' + k).
    dp = [[[0] * 27 for _ in range(n)] for _ in range(n)]

    for d in range(1, n):
      for i in range(n - d):
        for k in range(27):
          j = i + d
          if s[i] == s[j] and s[i] != chr(ord('a') + k):
            dp[i][j][k] = dp[i + 1][j - 1][ord(s[i]) - ord('a')] + 2
          else:
            dp[i][j][k] = max(dp[i + 1][j][k], dp[i][j - 1][k])

    return dp[0][n - 1][26]


# Link: https://leetcode.com/problems/longest-palindromic-subsequence-ii/description/
class Solution:
  def longestPalindromeSubseq(self, s: str) -> int:
    n = len(s)

    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> int:
      """
      Returns the length of LPS(s[i..j]), where the previous letter is
      ('a' + k).
      """
      if i >= j:
        return 0
      if s[i] == s[j] and s[i] != chr(ord('a') + k):
        return dp(i + 1, j - 1, ord(s[i]) - ord('a')) + 2
      return max(dp(i + 1, j, k), dp(i, j - 1, k))

    return dp(0, n - 1, 26)


# Link: https://leetcode.com/problems/detonate-the-maximum-bombs/description/
class Solution:
  def maximumDetonation(self, bombs: List[List[int]]) -> int:
    n = len(bombs)
    ans = 0
    graph = [[] for _ in range(n)]

    for i, (xi, yi, ri) in enumerate(bombs):
      for j, (xj, yj, rj) in enumerate(bombs):
        if i == j:
          continue
        if ri**2 >= (xi - xj)**2 + (yi - yj)**2:
          graph[i].append(j)

    def dfs(u: int, seen: Set[int]) -> None:
      for v in graph[u]:
        if v in seen:
          continue
        seen.add(v)
        dfs(v, seen)

    for i in range(n):
      seen = set([i])
      dfs(i, seen)
      ans = max(ans, len(seen))

    return ans


# Link: https://leetcode.com/problems/maximum-subtree-of-the-same-color/description/
class Solution:
  def maximumSubtreeSize(self, edges: List[List[int]], colors: List[int]) -> int:
    ans = 1
    tree = [[] for _ in range(len(colors))]

    for u, v in edges:
      tree[u].append(v)

    def dfs(u: int) -> int:
      """
      Returns the size of subtree of u if every node in the subtree has the same
      color. Otherwise, returns -1.
      """
      nonlocal ans
      res = 1
      for v in tree[u]:
        if colors[v] != colors[u]:
          res = -1
        # If any node in the subtree of v has a different color, the result of
        # the subtree of u will be -1 as well.
        subtreeSize = dfs(v)
        if subtreeSize == -1:
          res = -1
        elif res != -1:
          res += subtreeSize
      ans = max(ans, res)
      return res

    dfs(0)
    return ans


# Link: https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/description/
class Solution:
  def minSwaps(self, s: str) -> int:
    # Cancel out all the matched pairs, then we'll be left with ']]]..[[['.
    # The answer is ceil(# of unmatched pairs // 2).
    unmatched = 0

    for c in s:
      if c == '[':
        unmatched += 1
      elif unmatched > 0:  # c == ']' and there's a match.
        unmatched -= 1

    return (unmatched + 1) // 2


# Link: https://leetcode.com/problems/smallest-range-ii/description/
class Solution:
  def smallestRangeII(self, nums: List[int], k: int) -> int:
    nums.sort()

    ans = nums[-1] - nums[0]
    left = nums[0] + k
    right = nums[-1] - k

    for a, b in itertools.pairwise(nums):
      mini = min(left, b - k)
      maxi = max(right, a + k)
      ans = min(ans, maxi - mini)

    return ans


# Link: https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/description/
class Solution:
  def reverseParentheses(self, s: str) -> str:
    ans = []
    stack = []
    pair = {}

    for i, c in enumerate(s):
      if c == '(':
        stack.append(i)
      elif c == ')':
        j = stack.pop()
        pair[i] = j
        pair[j] = i

    i = 0
    d = 1
    while i < len(s):
      if s[i] in '()':
        i = pair[i]
        d = -d
      else:
        ans.append(s[i])
      i += d

    return ''.join(ans)


# Link: https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/description/
class Solution:
  def reverseParentheses(self, s: str) -> str:
    stack = []
    ans = []

    for c in s:
      if c == '(':
        stack.append(len(ans))
      elif c == ')':
        # Reverse the corresponding substring between ().
        j = stack.pop()
        ans[j:] = ans[j:][::-1]
      else:
        ans.append(c)

    return ''.join(ans)


# Link: https://leetcode.com/problems/maximum-subarray/description/
class Solution:
  def maxSubArray(self, nums: List[int]) -> int:
    ans = -math.inf
    summ = 0

    for num in nums:
      summ = max(num, summ + num)
      ans = max(ans, summ)

    return ans


# Link: https://leetcode.com/problems/maximum-subarray/description/
from dataclasses import dataclass


@dataclass(frozen=True)
class T:
  # the sum of the subarray starting from the first number
  maxSubarraySumLeft: int
  # the sum of the subarray ending in the last number
  maxSubarraySumRight: int
  maxSubarraySum: int
  summ: int


class Solution:
  def maxSubArray(self, nums: List[int]) -> int:
    def divideAndConquer(l: int, r: int) -> T:
      if l == r:
        return T(nums[l], nums[l], nums[l], nums[l])

      m = (l + r) // 2
      t1 = divideAndConquer(l, m)
      t2 = divideAndConquer(m + 1, r)

      maxSubarraySumLeft = max(t1.maxSubarraySumLeft,
                               t1.summ + t2.maxSubarraySumLeft)
      maxSubarraySumRight = max(
          t1.maxSubarraySumRight + t2.summ, t2.maxSubarraySumRight)
      maxSubarraySum = max(t1.maxSubarraySumRight +
                           t2.maxSubarraySumLeft, t1.maxSubarraySum, t2.maxSubarraySum)
      summ = t1.summ + t2.summ
      return T(maxSubarraySumLeft, maxSubarraySumRight, maxSubarraySum, summ)

    return divideAndConquer(0, len(nums) - 1).maxSubarraySum


# Link: https://leetcode.com/problems/maximum-subarray/description/
class Solution:
  def maxSubArray(self, nums: List[int]) -> int:
    # dp[i] := the maximum sum subarray ending in i
    dp = [0] * len(nums)

    dp[0] = nums[0]
    for i in range(1, len(nums)):
      dp[i] = max(nums[i], dp[i - 1] + nums[i])

    return max(dp)


# Link: https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/description/
from abc import ABC, abstractmethod
"""
This is the interface for the expression tree Node.
You should not remove it, and you can define some classes to implement it.
"""


class Node(ABC):
  @abstractmethod
  # define your fields here
  def evaluate(self) -> int:
    pass


class ExpNode(Node):
  op = {
      '+': lambda a, b: a + b,
      '-': lambda a, b: a - b,
      '*': lambda a, b: a * b,
      '/': lambda a, b: int(a / b),
  }

  def __init__(self, val: str, left: Optional['ExpNode'], right: Optional['ExpNode']):
    self.val = val
    self.left = left
    self.right = right

  def evaluate(self) -> int:
    if not self.left and not self.right:
      return int(self.val)
    return ExpNode.op[self.val](self.left.evaluate(), self.right.evaluate())


"""
This is the TreeBuilder class.
You can treat it as the driver code that takes the postinfix input
and returns the expression tree represnting it as a Node.
"""


class TreeBuilder(object):
  def buildTree(self, postfix: List[str]) -> 'Node':
    stack: List[Optional[ExpNode]] = []

    for val in postfix:
      if val in '+-*/':
        right = stack.pop()
        left = stack.pop()
        stack.append(ExpNode(val, left, right))
      else:
        stack.append(ExpNode(val, None, None))

    return stack.pop()


# Link: https://leetcode.com/problems/rotate-image/description/
class Solution:
  def rotate(self, matrix: List[List[int]]) -> None:
    for min in range(len(matrix) // 2):
      max = len(matrix) - min - 1
      for i in range(min, max):
        offset = i - min
        top = matrix[min][i]
        matrix[min][i] = matrix[max - offset][min]
        matrix[max - offset][min] = matrix[max][max - offset]
        matrix[max][max - offset] = matrix[i][max]
        matrix[i][max] = top


# Link: https://leetcode.com/problems/rotate-image/description/
class Solution:
  def rotate(self, matrix: List[List[int]]) -> None:
    matrix.reverse()

    for i in range(len(matrix)):
      for j in range(i + 1, len(matrix)):
        matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


# Link: https://leetcode.com/problems/subarray-sums-divisible-by-k/description/
class Solution:
  def subarraysDivByK(self, nums: List[int], k: int) -> int:
    ans = 0
    prefix = 0
    count = [0] * k
    count[0] = 1

    for num in nums:
      prefix = (prefix + num % k + k) % k
      ans += count[prefix]
      count[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/path-with-maximum-gold/description/
class Solution:
  def getMaximumGold(self, grid: List[List[int]]) -> int:
    def dfs(i: int, j: int) -> int:
      if i < 0 or j < 0 or i == len(grid) or j == len(grid[0]):
        return 0
      if grid[i][j] == 0:
        return 0

      gold = grid[i][j]
      grid[i][j] = 0  # Mark as visited.
      maxPath = max(dfs(i + 1, j), dfs(i - 1, j),
                    dfs(i, j + 1), dfs(i, j - 1))
      grid[i][j] = gold
      return gold + maxPath

    return max(dfs(i, j)
               for i in range(len(grid))
               for j in range(len(grid[0])))


# Link: https://leetcode.com/problems/find-champion-ii/description/
class Solution:
  def findChampion(self, n: int, edges: List[List[int]]) -> int:
    inDegrees = [0] * n

    for _, v in edges:
      inDegrees[v] += 1

    return -1 if inDegrees.count(0) > 1 \
        else inDegrees.index(0)


# Link: https://leetcode.com/problems/apply-operations-to-make-two-strings-equal/description/
class Solution:
  def minOperations(self, s1: str, s2: str, x: int) -> int:
    diffIndices = [i for i, (a, b) in enumerate(zip(s1, s2))
                   if a != b]
    if not diffIndices:
      return 0
    # It's impossible to make two strings equal if there are odd number of
    # differences.
    if len(diffIndices) & 1:
      return -1

    @functools.lru_cache(None)
    def dp(i: int) -> int:
      """Returns the minimum cost to correct diffIndices[i..n)."""
      if i == len(diffIndices):
        return 0
      if i == len(diffIndices) - 1:
        return x / 2
      return min(dp(i + 1) + x / 2,
                 dp(i + 2) + diffIndices[i + 1] - diffIndices[i])

    return int(dp(0))


# Link: https://leetcode.com/problems/apply-operations-to-make-two-strings-equal/description/
class Solution:
  def minOperations(self, s1: str, s2: str, x: int) -> int:
    diffIndices = [i for i, (a, b) in enumerate(zip(s1, s2))
                   if a != b]
    if not diffIndices:
      return 0
    # It's impossible to make two strings equal if there are odd number of
    # differences.
    if len(diffIndices) & 1:
      return -1

    #         dp := the minimum cost to correct diffIndices[i:]
    #     dpNext := the minimum cost to correct diffIndices[i + 1:]
    # dpNextNext := the minimum cost to correct diffIndices[i + 2:]
    dpNext = x / 2
    dpNextNext = 0

    for i in reversed(range(len(diffIndices) - 1)):
      dp = min(dpNext + x / 2,
               dpNextNext + diffIndices[i + 1] - diffIndices[i])
      dpNextNext = dpNext
      dpNext = dp

    return int(dp)


# Link: https://leetcode.com/problems/apply-operations-to-make-two-strings-equal/description/
class Solution:
  def minOperations(self, s1: str, s2: str, x: int) -> int:
    diffIndices = [i for i, (a, b) in enumerate(zip(s1, s2))
                   if a != b]
    if not diffIndices:
      return 0
    # It's impossible to make two strings equal if there are odd number of
    # differences.
    if len(diffIndices) & 1:
      return -1

    # dp[i] := the minimum cost to correct diffIndices[i:]
    dp = [math.inf] * len(diffIndices) + [0]
    dp[-2] = x / 2

    for i in reversed(range(len(diffIndices) - 1)):
      dp[i] = min(dp[i + 1] + x / 2,
                  dp[i + 2] + diffIndices[i + 1] - diffIndices[i])

    return int(dp[0])


# Link: https://leetcode.com/problems/remove-duplicates-from-an-unsorted-linked-list/description/
class Solution:
  def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
    dummy = ListNode(0, head)
    count = collections.Counter()

    curr = head
    while curr:
      count[curr.val] += 1
      curr = curr.next

    curr = dummy

    while curr:
      while curr.next and curr.next.val in count and count[curr.next.val] > 1:
        curr.next = curr.next.next
      curr = curr.next

    return dummy.next


# Link: https://leetcode.com/problems/maximum-xor-for-each-query/description/
class Solution:
  def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
    max = (1 << maximumBit) - 1
    ans = []
    xors = 0

    for num in nums:
      xors ^= num
      ans.append(xors ^ max)

    return ans[::-1]


# Link: https://leetcode.com/problems/subrectangle-queries/description/
class SubrectangleQueries:
  def __init__(self, rectangle: List[List[int]]):
    self.rectangle = rectangle
    self.updates = []

  def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
    self.updates.append((row1, col1, row2, col2, newValue))

  def getValue(self, row: int, col: int) -> int:
    for r1, c1, r2, c2, v in reversed(self.updates):
      if r1 <= row <= r2 and c1 <= col <= c2:
        return v
    return self.rectangle[row][col]


# Link: https://leetcode.com/problems/subsets/description/
class Solution:
  def subsets(self, nums: List[int]) -> List[List[int]]:
    ans = []

    def dfs(s: int, path: List[int]) -> None:
      ans.append(path)

      for i in range(s, len(nums)):
        dfs(i + 1, path + [nums[i]])

    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/apply-discount-to-prices/description/
class Solution:
  def discountPrices(self, sentence: str, discount: int) -> str:
    kPrecision = 2
    ans = []

    for word in sentence.split():
      if word[0] == '$' and len(word) > 1:
        digits = word[1:]
        if all(digit.isdigit() for digit in digits):
          val = float(digits) * (100 - discount) / 100
          s = f'{val:.2f}'
          trimmed = s[:s.index('.') + kPrecision + 1]
          ans.append('$' + trimmed)
        else:
          ans.append(word)
      else:
        ans.append(word)

    return ' '.join(ans)


# Link: https://leetcode.com/problems/rearrange-array-elements-by-sign/description/
class Solution:
  def rearrangeArray(self, nums: List[int]) -> List[int]:
    ans = []
    pos = []
    neg = []

    for num in nums:
      (pos if num > 0 else neg).append(num)

    for p, n in zip(pos, neg):
      ans += [p, n]

    return ans


# Link: https://leetcode.com/problems/number-of-good-ways-to-split-a-string/description/
class Solution:
  def numSplits(self, s: str) -> int:
    n = len(s)
    ans = 0
    seen = set()
    prefix = [0] * n
    suffix = [0] * n

    for i in range(n):
      seen.add(s[i])
      prefix[i] = len(seen)

    seen.clear()

    for i in reversed(range(n)):
      seen.add(s[i])
      suffix[i] = len(seen)

    for i in range(n - 1):
      if prefix[i] == suffix[i + 1]:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three/description/
class Solution:
  def checkPowersOfThree(self, n: int) -> bool:
    while n > 1:
      n, r = divmod(n, 3)
      if r == 2:
        return False
    return True


# Link: https://leetcode.com/problems/decode-ways/description/
class Solution:
  def numDecodings(self, s: str) -> int:
    n = len(s)
    # dp[i] := the number of ways to decode s[i..n)
    dp = [0] * n + [1]

    def isValid(a: str, b=None) -> bool:
      if b:
        return a == '1' or a == '2' and b < '7'
      return a != '0'

    if isValid(s[-1]):
      dp[n - 1] = 1

    for i in reversed(range(n - 1)):
      if isValid(s[i]):
        dp[i] += dp[i + 1]
      if isValid(s[i], s[i + 1]):
        dp[i] += dp[i + 2]

    return dp[0]


# Link: https://leetcode.com/problems/friends-of-appropriate-ages/description/
class Solution:
  def numFriendRequests(self, ages: List[int]) -> int:
    ans = 0
    count = [0] * 121

    for age in ages:
      count[age] += 1

    for i in range(15, 121):
      ans += count[i] * (count[i] - 1)

    for i in range(15, 121):
      for j in range(i // 2 + 8, i):
        ans += count[i] * count[j]

    return ans


# Link: https://leetcode.com/problems/longest-common-subsequence/description/
class Solution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    m = len(text1)
    n = len(text2)
    # dp[i][j] := the length of LCS(text1[0..i), text2[0..j))
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        dp[i + 1][j + 1] = \
            1 + dp[i][j] if text1[i] == text2[j] \
            else max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n]


# Link: https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/
class Solution:
  def pairSum(self, head: Optional[ListNode]) -> int:
    def reverseList(head: ListNode) -> ListNode:
      prev = None
      while head:
        next = head.next
        head.next = prev
        prev = head
        head = next
      return prev

    ans = 0
    slow = head
    fast = head

    # `slow` points to the start of the second half.
    while fast and fast.next:
      slow = slow.next
      fast = fast.next.next

    # `tail` points to the end of the reversed second half.
    tail = reverseList(slow)

    while tail:
      ans = max(ans, head.val + tail.val)
      head = head.next
      tail = tail.next

    return ans


# Link: https://leetcode.com/problems/design-twitter/description/
class Twitter:
  def __init__(self):
    self.timer = itertools.count(step=-1)
    self.tweets = collections.defaultdict(deque)
    self.followees = collections.defaultdict(set)

  def postTweet(self, userId: int, tweetId: int) -> None:
    self.tweets[userId].appendleft((next(self.timer), tweetId))
    if len(self.tweets[userId]) > 10:
      self.tweets[userId].pop()

  def getNewsFeed(self, userId: int) -> List[int]:
    tweets = list(heapq.merge(
        *(self.tweets[followee] for followee in self.followees[userId] | {userId})))
    return [tweetId for _, tweetId in tweets[:10]]

  def follow(self, followerId: int, followeeId: int) -> None:
    self.followees[followerId].add(followeeId)

  def unfollow(self, followerId: int, followeeId: int) -> None:
    self.followees[followerId].discard(followeeId)


# Link: https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/description/
class Solution:
  def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def dfs(root: Optional[TreeNode], path: int) -> None:
      nonlocal ans
      if not root:
        return
      if not root.left and not root.right:
        path ^= 1 << root.val
        if path & (path - 1) == 0:
          ans += 1
        return

      dfs(root.left, path ^ 1 << root.val)
      dfs(root.right, path ^ 1 << root.val)

    dfs(root, 0)
    return ans


# Link: https://leetcode.com/problems/minimize-the-maximum-of-two-arrays/description/
class Solution:
  def minimizeSet(self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int) -> int:
    divisorLcm = math.lcm(divisor1, divisor2)
    l = 0
    r = 2**31 - 1

    # True if we can take uniqueCnt1 integers from [1..m] to arr1 and take
    # uniqueCnt2 integers from [1..m] to arr2.
    def isPossible(m: int) -> bool:
      cnt1 = m - m // divisor1
      cnt2 = m - m // divisor2
      totalCnt = m - m // divisorLcm
      return cnt1 >= uniqueCnt1 and cnt2 >= uniqueCnt2 and \
          totalCnt >= uniqueCnt1 + uniqueCnt2

    while l < r:
      m = (l + r) // 2
      if isPossible(m):
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/fraction-to-recurring-decimal/description/
class Solution:
  def fractionToDecimal(self, numerator: int, denominator: int) -> str:
    if numerator == 0:
      return '0'

    ans = ''

    if (numerator < 0) ^ (denominator < 0):
      ans += '-'

    numerator = abs(numerator)
    denominator = abs(denominator)
    ans += str(numerator // denominator)

    if numerator % denominator == 0:
      return ans

    ans += '.'
    dict = {}

    remainder = numerator % denominator
    while remainder:
      if remainder in dict:
        ans = ans[:dict[remainder]] + '(' + ans[dict[remainder]:] + ')'
        break
      dict[remainder] = len(ans)
      remainder *= 10
      ans += str(remainder // denominator)
      remainder %= denominator

    return ans


# Link: https://leetcode.com/problems/optimal-partition-of-string/description/
class Solution:
  def partitionString(self, s: str) -> int:
    ans = 1
    used = 0

    for c in s:
      i = ord(c) - ord('a')
      if used >> i & 1:
        used = 1 << i
        ans += 1
      else:
        used |= 1 << i

    return ans


# Link: https://leetcode.com/problems/distribute-candies-among-children-ii/description/
class Solution:
  def distributeCandies(self, n: int, limit: int) -> int:
    def ways(n: int) -> int:
      """Returns the number of ways to distribute n candies to 3 children."""
      if n < 0:
        return 0
      # Stars and bars method:
      # e.g. '**|**|*' means to distribute 5 candies to 3 children, where
      # stars (*) := candies and bars (|) := dividers between children.
      return math.comb(n + 2, 2)

    limitPlusOne = limit + 1
    oneChildExceedsLimit = ways(n - limitPlusOne)
    twoChildrenExceedLimit = ways(n - 2 * limitPlusOne)
    threeChildrenExceedLimit = ways(n - 3 * limitPlusOne)
    # Principle of Inclusion-Exclusion (PIE)
    return ways(n) \
        - 3 * oneChildExceedsLimit \
        + 3 * twoChildrenExceedLimit \
        - threeChildrenExceedLimit


# Link: https://leetcode.com/problems/distinct-prime-factors-of-product-of-array/description/
class Solution:
  def distinctPrimeFactors(self, nums: List[int]) -> int:
    primes = set()

    for num in nums:
      self._addPrimeFactors(primes, num)

    return len(primes)

  def _addPrimeFactors(self, primes: Set[int], num: int) -> None:
    for divisor in range(2, num + 1):
      if num % divisor == 0:
        primes.add(divisor)
        while num % divisor == 0:
          num //= divisor


# Link: https://leetcode.com/problems/single-number-iii/description/
class Solution:
  def singleNumber(self, nums: List[int]) -> List[int]:
    xors = functools.reduce(operator.xor, nums)
    lowbit = xors & -xors
    ans = [0, 0]

    # Seperate `nums` into two groups by `lowbit`.
    for num in nums:
      if num & lowbit:
        ans[0] ^= num
      else:
        ans[1] ^= num

    return ans


# Link: https://leetcode.com/problems/count-pairs-in-two-arrays/description/
class Solution:
  def countPairs(self, nums1: List[int], nums2: List[int]) -> int:
    ans = 0
    A = sorted([x - y for x, y in zip(nums1, nums2)])

    for i, a in enumerate(A):
      index = bisect_left(A, -a + 1)
      ans += len(A) - max(i + 1, index)

    return ans


# Link: https://leetcode.com/problems/insert-greatest-common-divisors-in-linked-list/description/
class Solution:
  def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
    curr = head
    while curr.next:
      inserted = ListNode(math.gcd(curr.val, curr.next.val), curr.next)
      curr.next = inserted
      curr = inserted.next
    return head


# Link: https://leetcode.com/problems/sort-the-students-by-their-kth-score/description/
class Solution:
  def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
    return sorted(score, key=lambda a: -a[k])


# Link: https://leetcode.com/problems/beautiful-array/description/
class Solution:
  def beautifulArray(self, n: int) -> List[int]:
    A = [i for i in range(1, n + 1)]

    def partition(l: int, r: int, mask: int) -> int:
      nextSwapped = l
      for i in range(l, r + 1):
        if A[i] & mask:
          A[i], A[nextSwapped] = A[nextSwapped], A[i]
          nextSwapped += 1
      return nextSwapped - 1

    def divide(l: int, r: int, mask: int) -> None:
      if l >= r:
        return
      m = partition(l, r, mask)
      divide(l, m, mask << 1)
      divide(m + 1, r, mask << 1)

    divide(0, n - 1, 1)
    return A


# Link: https://leetcode.com/problems/difference-between-ones-and-zeros-in-row-and-column/description/
class Solution:
  def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
    m = len(grid)
    n = len(grid[0])
    ans = [[0] * n for _ in range(m)]
    onesRow = [row.count(1) for row in grid]
    onesCol = [col.count(1) for col in zip(*grid)]

    for i in range(m):
      for j in range(n):
        ans[i][j] = onesRow[i] + onesCol[j] - \
            (n - onesRow[i]) - (m - onesCol[j])

    return ans


# Link: https://leetcode.com/problems/balance-a-binary-search-tree/description/
class Solution:
  def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    nums = []

    def inorder(root: Optional[TreeNode]) -> None:
      if not root:
        return
      inorder(root.left)
      nums.append(root.val)
      inorder(root.right)

    inorder(root)

    # Same as 108. Convert Sorted Array to Binary Search Tree
    def build(l: int, r: int) -> Optional[TreeNode]:
      if l > r:
        return None
      m = (l + r) // 2
      return TreeNode(nums[m],
                      build(l, m - 1),
                      build(m + 1, r))

    return build(0, len(nums) - 1)


# Link: https://leetcode.com/problems/count-servers-that-communicate/description/
class Solution:
  def countServers(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    ans = 0
    rows = [0] * m
    cols = [0] * n

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1:
          rows[i] += 1
          cols[j] += 1

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1 and (rows[i] > 1 or cols[j] > 1):
          ans += 1

    return ans


# Link: https://leetcode.com/problems/basic-calculator-ii/description/
class Solution:
  def calculate(self, s: str) -> int:
    ans = 0
    prevNum = 0
    currNum = 0
    op = '+'

    for i, c in enumerate(s):
      if c.isdigit():
        currNum = currNum * 10 + int(c)
      if not c.isdigit() and c != ' ' or i == len(s) - 1:
        if op == '+' or op == '-':
          ans += prevNum
          prevNum = currNum if op == '+' else -currNum
        elif op == '*':
          prevNum = prevNum * currNum
        elif op == '/':
          if prevNum < 0:
            prevNum = math.ceil(prevNum / currNum)
          else:
            prevNum = prevNum // currNum
        op = c
        currNum = 0

    return ans + prevNum


# Link: https://leetcode.com/problems/basic-calculator-ii/description/
class Solution:
  def calculate(self, s: str) -> int:
    ans = 0
    prevNum = 0
    currNum = 0
    op = '+'

    for i, c in enumerate(s):
      if c.isdigit():
        currNum = currNum * 10 + int(c)
      if not c.isdigit() and c != ' ' or i == len(s) - 1:
        if op == '+' or op == '-':
          ans += prevNum
          prevNum = (currNum if op == '+' else -currNum)
        elif op == '*':
          prevNum *= currNum
        elif op == '/':
          prevNum = int(prevNum / currNum)
        op = c
        currNum = 0

    return ans + prevNum


# Link: https://leetcode.com/problems/process-tasks-using-servers/description/
class Solution:
  def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
    ans = []
    free = []  # (weight, index, freeTime)
    used = []  # (freeTime, weight, index)

    for i, weight in enumerate(servers):
      heapq.heappush(free, (weight, i, 0))

    for i, executionTime in enumerate(tasks):  # i := the current time
      # Poll all servers that'll be free at time i.
      while used and used[0][0] <= i:
        curr = heapq.heappop(used)
        heapq.heappush(free, (curr[1], curr[2], curr[0]))
      if free:
        curr = heapq.heappop(free)
        ans.append(curr[1])
        heapq.heappush(used, (i + executionTime, curr[0], curr[1]))
      else:
        curr = heapq.heappop(used)
        ans.append(curr[2])
        heapq.heappush(used, (curr[0] + executionTime, curr[1], curr[2]))

    return ans


# Link: https://leetcode.com/problems/mirror-reflection/description/
class Solution:
  def mirrorReflection(self, p: int, q: int) -> int:
    while p % 2 == 0 and q % 2 == 0:
      p //= 2
      q //= 2

    if p % 2 == 0:
      return 2
    if q % 2 == 0:
      return 0
    return 1


# Link: https://leetcode.com/problems/number-of-substrings-with-fixed-ratio/description/
class Solution:
  def fixedRatio(self, s: str, num1: int, num2: int) -> int:
    # Let x := the number of 0s and y := the number of 1s in the subarray.
    # We want x : y = num1 : num2, so our goal is to find number of subarrays
    # with x * num2 - y * num1 = 0. To achieve this, we can use a prefix count
    # map to record the count of the running x * num2 - y * num1. If the
    # running x * num2 - y * num1 = prefix, then add count[prefix] to the
    # `ans`.
    ans = 0
    prefix = 0
    prefixCount = collections.Counter({0: 1})

    for c in s:
      if c == '0':
        prefix += num2
      else:  # c == '1'
        prefix -= num1
      ans += prefixCount[prefix]
      prefixCount[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/description/
class Solution:
  def canBeValid(self, s: str, locked: str) -> bool:
    if len(s) & 1:
      return False

    def check(s: str, locked: str, isForward: bool) -> bool:
      changeable = 0
      l = 0
      r = 0

      for c, lock in zip(s, locked):
        if lock == '0':
          changeable += 1
        elif c == '(':
          l += 1
        else:  # c == ')'
          r += 1
        if isForward and changeable + l - r < 0:
          return False
        if not isForward and changeable + r - l < 0:
          return False

      return True

    return check(s, locked, True) and check(s[::-1], locked[::-1], False)


# Link: https://leetcode.com/problems/parallel-courses/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
    graph = [[] for _ in range(n)]
    states = [State.kInit] * n
    depth = [1] * n

    for u, v in relations:
      graph[u - 1].append(v - 1)

    def hasCycle(u: int) -> bool:
      if states[u] == State.kVisiting:
        return True
      if states[u] == State.kVisited:
        return False

      states[u] = State.kVisiting
      for v in graph[u]:
        if hasCycle(v):
          return True
        depth[u] = max(depth[u], 1 + depth[v])
      states[u] = State.kVisited

      return False

    if any(hasCycle(i) for i in range(n)):
      return -1
    return max(depth)


# Link: https://leetcode.com/problems/parallel-courses/description/
class Solution:
  def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
    ans = 0
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n

    # Build the graph.
    for u, v in relations:
      graph[u - 1].append(v - 1)
      inDegrees[v - 1] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    while q:
      for _ in range(len(q)):
        u = q.popleft()
        n -= 1
        for v in graph[u]:
          inDegrees[v] -= 1
          if inDegrees[v] == 0:
            q.append(v)
      ans += 1

    return ans if n == 0 else -1


# Link: https://leetcode.com/problems/binary-string-with-substrings-representing-1-to-n/description/
class Solution:
  def queryString(self, s: str, n: int) -> bool:
    if n > 1511:
      return False

    for i in range(n, n // 2, -1):
      if format(i, 'b') not in s:
        return False

    return True


# Link: https://leetcode.com/problems/integer-replacement/description/
class Solution:
  def integerReplacement(self, n: int) -> int:
    ans = 0

    while n > 1:
      if (n & 1) == 0:  # `n` ends in 0.
        n >>= 1
      elif n == 3 or ((n >> 1) & 1) == 0:  # `n` = 3 or ends in 0b01.
        n -= 1
      else:  # `n` ends in 0b11.
        n += 1
      ans += 1

    return ans


# Link: https://leetcode.com/problems/number-of-pairs-of-interchangeable-rectangles/description/
class Solution:
  def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
    ratioCount = collections.Counter()

    def gcd(a: int, b: int) -> int:
      return a if b == 0 else gcd(b, a % b)

    for width, height in rectangles:
      d = gcd(width, height)
      ratioCount[(width // d, height // d)] += 1

    return sum(c * (c - 1) // 2 for c in ratioCount.values())


# Link: https://leetcode.com/problems/number-of-pairs-of-interchangeable-rectangles/description/
class Solution:
  def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
    ratioCount = collections.Counter()

    for width, height in rectangles:
      ratioCount[width / height] += 1

    return sum(c * (c - 1) // 2 for c in ratioCount.values())


# Link: https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/description/
class Solution:
  def minOperations(self, nums: List[int], k: int) -> int:
    ans = 0
    minHeap = nums.copy()
    heapq.heapify(minHeap)

    while len(minHeap) > 1 and minHeap[0] < k:
      x = heapq.heappop(minHeap)
      y = heapq.heappop(minHeap)
      heapq.heappush(minHeap, min(x, y) * 2 + max(x, y))
      ans += 1

    return ans


# Link: https://leetcode.com/problems/sort-list/description/
class Solution:
  def sortList(self, head: ListNode) -> ListNode:
    def split(head: ListNode, k: int) -> ListNode:
      while k > 1 and head:
        head = head.next
        k -= 1
      rest = head.next if head else None
      if head:
        head.next = None
      return rest

    def merge(l1: ListNode, l2: ListNode) -> tuple:
      dummy = ListNode(0)
      tail = dummy

      while l1 and l2:
        if l1.val > l2.val:
          l1, l2 = l2, l1
        tail.next = l1
        l1 = l1.next
        tail = tail.next
      tail.next = l1 if l1 else l2
      while tail.next:
        tail = tail.next

      return dummy.next, tail

    length = 0
    curr = head
    while curr:
      length += 1
      curr = curr.next

    dummy = ListNode(0, head)

    k = 1
    while k < length:
      curr = dummy.next
      tail = dummy
      while curr:
        l = curr
        r = split(l, k)
        curr = split(r, k)
        mergedHead, mergedTail = merge(l, r)
        tail.next = mergedHead
        tail = mergedTail
      k *= 2

    return dummy.next


# Link: https://leetcode.com/problems/shopping-offers/description/
class Solution:
  def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
    def dfs(s: int) -> int:
      ans = 0
      for i, need in enumerate(needs):
        ans += need * price[i]

      for i in range(s, len(special)):
        offer = special[i]
        if all(offer[j] <= need for j, need in enumerate(needs)):
          # Use the special[i].
          for j in range(len(needs)):
            needs[j] -= offer[j]
          ans = min(ans, offer[-1] + dfs(i))
          # Unuse the special[i] (backtracking).
          for j in range(len(needs)):
            needs[j] += offer[j]

      return ans

    return dfs(0)


# Link: https://leetcode.com/problems/maximum-coins-heroes-can-collect/description/
class Solution:
  def maximumCoins(self, heroes: List[int], monsters: List[int], coins: List[int]) -> List[int]:
    monsterAndCoins = sorted(list(zip(monsters, coins)))
    coinsPrefix = [0] + \
        list(itertools.accumulate(coin for _, coin in monsterAndCoins))
    return [coinsPrefix[self._firstGreaterEqual(monsterAndCoins, hero)] for hero in heroes]

  def _firstGreaterEqual(self, monsterAndCoins: List[tuple[int, int]], hero: int) -> int:
    l, r = 0, len(monsterAndCoins)
    while l < r:
      m = (l + r) // 2
      if monsterAndCoins[m][0] > hero:
        r = m
      else:
        l = m + 1
    return l


# Link: https://leetcode.com/problems/rectangle-area/description/
class Solution:
  def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
    x = min(C, G) - max(A, E) if max(A, E) < min(C, G) else 0
    y = min(D, H) - max(B, F) if max(B, F) < min(D, H) else 0
    return (C - A) * (D - B) + (G - E) * (H - F) - x * y


# Link: https://leetcode.com/problems/powerful-integers/description/
class Solution:
  def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
    xs = {x**i for i in range(20) if x**i < bound}
    ys = {y**i for i in range(20) if y**i < bound}
    return list({i + j for i in xs for j in ys if i + j <= bound})


# Link: https://leetcode.com/problems/course-schedule-iv/description/
class Solution:
  def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
    ans = []
    # isPrerequisite[i][j] := True if course i is a prerequisite of course j.
    isPrerequisite = [[False] * numCourses for _ in range(numCourses)]

    for u, v in prerequisites:
      isPrerequisite[u][v] = True

    for k in range(numCourses):
      for i in range(numCourses):
        for j in range(numCourses):
          isPrerequisite[i][j] = isPrerequisite[i][j] or \
              (isPrerequisite[i][k] and isPrerequisite[k][j])

    return [isPrerequisite[u][v] for u, v in queries]


# Link: https://leetcode.com/problems/course-schedule-iv/description/
class Solution:
  def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
    graph = [[] for _ in range(numCourses)]
    # isPrerequisite[i][j] := True if course i is a prerequisite of course j.
    isPrerequisite = [[False] * numCourses for _ in range(numCourses)]

    for u, v in prerequisites:
      graph[u].append(v)

    # DFS from every course.
    for i in range(numCourses):
      self._dfs(graph, i, isPrerequisite[i])

    return [isPrerequisite[u][v] for u, v in queries]

  def _dfs(self, graph: List[List[int]], u: int, used: List[bool]) -> None:
    for v in graph[u]:
      if used[v]:
        continue
      used[v] = True
      self._dfs(graph, v, used)


# Link: https://leetcode.com/problems/statistics-from-a-large-sample/description/
class Solution:
  def sampleStats(self, count: List[int]) -> List[float]:
    minimum = next((i for i, num in enumerate(count) if num), None)
    maximum = next((i for i, num in reversed(
        list(enumerate(count))) if num), None)
    n = sum(count)
    mean = sum(i * c / n for i, c in enumerate(count))
    mode = count.index(max(count))

    numCount = 0
    leftMedian = 0
    for i, c in enumerate(count):
      numCount += c
      if numCount >= n / 2:
        leftMedian = i
        break

    numCount = 0
    rightMedian = 0
    for i, c in reversed(list(enumerate(count))):
      numCount += c
      if numCount >= n / 2:
        rightMedian = i
        break

    return [minimum, maximum, mean, (leftMedian + rightMedian) / 2, mode]


# Link: https://leetcode.com/problems/minesweeper/description/
class Solution:
  def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
    i, j = click
    if board[i][j] == 'M':
      board[i][j] = 'X'
      return board

    dirs = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1))

    def getMinesCount(i: int, j: int) -> int:
      minesCount = 0
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == len(board) or y < 0 or y == len(board[0]):
          continue
        if board[x][y] == 'M':
          minesCount += 1
      return minesCount

    def dfs(i: int, j: int) -> None:
      if i < 0 or i == len(board) or j < 0 or j == len(board[0]):
        return
      if board[i][j] != 'E':
        return

      minesCount = getMinesCount(i, j)
      board[i][j] = 'B' if minesCount == 0 else str(minesCount)

      if minesCount == 0:
        for dx, dy in dirs:
          dfs(i + dx, j + dy)

    dfs(i, j)
    return board


# Link: https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/description/
class Solution:
  def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
    for i, d in enumerate(sorted([c - r for c, r in zip(capacity, rocks)])):
      if d > additionalRocks:
        return i
      additionalRocks -= d
    return len(capacity)


# Link: https://leetcode.com/problems/substrings-that-begin-and-end-with-the-same-letter/description/
class Solution:
  def numberOfSubstrings(self, s: str) -> int:
    ans = 0
    count = collections.Counter()

    for c in s:
      ans += count[c] + 1
      count[c] += 1

    return ans


# Link: https://leetcode.com/problems/find-if-array-can-be-sorted/description/
class Solution:
  def canSortArray(self, nums: List[int]) -> int:
    # Divide the array into distinct segments where each segment is comprised
    # of consecutive elements sharing an equal number of set bits. Ensure that
    # for each segment, when moving from left to right, the maximum of a
    # preceding segment is less than the minimum of the following segment.
    prevSetBits = 0
    prevMax = -math.inf  # the maximum of the previous segment
    currMax = -math.inf  # the maximum of the current segment
    currMin = math.inf   # the minimum of the current segment

    for num in nums:
      setBits = num.bit_count()
      if setBits != prevSetBits:  # Start a new segment.
        if prevMax > currMin:
          return False
        prevSetBits = setBits
        prevMax = currMax
        currMax = num
        currMin = num
      else:  # Continue with the current segment.
        currMax = max(currMax, num)
        currMin = min(currMin, num)

    return prevMax <= currMin


# Link: https://leetcode.com/problems/maximum-length-of-subarray-with-positive-product/description/
class Solution:
  def getMaxLen(self, nums: List[int]) -> int:
    ans = 0
    # the maximum length of subarrays ending in `num` with a negative product
    neg = 0
    # the maximum length of subarrays ending in `num` with a positive product
    pos = 0

    for num in nums:
      pos = 0 if num == 0 else pos + 1
      neg = 0 if num == 0 or neg == 0 else neg + 1
      if num < 0:
        pos, neg = neg, pos
      ans = max(ans, pos)

    return ans


# Link: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
class Solution:
  def twoSum(self, numbers: List[int], target: int) -> List[int]:
    l = 0
    r = len(numbers) - 1

    while l < r:
      summ = numbers[l] + numbers[r]
      if summ == target:
        return [l + 1, r + 1]
      if summ < target:
        l += 1
      else:
        r -= 1


# Link: https://leetcode.com/problems/maximal-score-after-applying-k-operations/description/
class Solution:
  def maxKelements(self, nums: List[int], k: int) -> int:
    ans = 0
    maxHeap = [-num for num in nums]
    heapq.heapify(maxHeap)

    for _ in range(k):
      num = -heapq.heappop(maxHeap)
      ans += num
      heapq.heappush(maxHeap, -math.ceil(num / 3))

    return ans


# Link: https://leetcode.com/problems/maximum-subsequence-score/description/
class Solution:
  # Same as 1383. Maximum Performance of a Team
  def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
    ans = 0
    summ = 0
    # (nums2[i], nums1[i]) sorted by nums2[i] in descending order.
    A = sorted([(num2, num1)
               for num1, num2 in zip(nums1, nums2)], reverse=True)
    minHeap = []

    for num2, num1 in A:
      heapq.heappush(minHeap, num1)
      summ += num1
      if len(minHeap) > k:
        summ -= heapq.heappop(minHeap)
      if len(minHeap) == k:
        ans = max(ans, summ * num2)

    return ans


# Link: https://leetcode.com/problems/can-make-palindrome-from-substring/description/
class Solution:
  def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
    def ones(x):
      return bin(x).count('1')

    dp = [0] * (len(s) + 1)

    for i in range(1, len(s) + 1):
      dp[i] = dp[i - 1] ^ 1 << ord(s[i - 1]) - ord('a')

    return [
        ones(dp[right + 1] ^ dp[left]) // 2 <= k
        for left, right, k in queries
    ]


# Link: https://leetcode.com/problems/reorganize-string/description/
class Solution:
  def reorganizeString(self, s: str) -> str:
    count = collections.Counter(s)
    if max(count.values()) > (len(s) + 1) // 2:
      return ''

    ans = []
    maxHeap = [(-freq, c) for c, freq in count.items()]
    heapq.heapify(maxHeap)
    prevFreq = 0
    prevChar = '@'

    while maxHeap:
      # Get the letter with the maximum frequency.
      freq, c = heapq.heappop(maxHeap)
      ans.append(c)
      # Add the previous letter back s.t. any two adjacent characters are not
      # the same.
      if prevFreq < 0:
        heapq.heappush(maxHeap, (prevFreq, prevChar))
      prevFreq = freq + 1
      prevChar = c

    return ''.join(ans)


# Link: https://leetcode.com/problems/reorganize-string/description/
class Solution:
  def reorganizeString(self, s: str) -> str:
    n = len(s)
    count = collections.Counter(s)
    maxCount = max(count.values())

    if maxCount > (n + 1) // 2:
      return ''

    if maxCount == (n + 1) // 2:
      maxLetter = max(count, key=count.get)
      ans = [maxLetter if i % 2 == 0 else '' for i in range(n)]
      del count[maxLetter]
      i = 1
    else:
      ans = [''] * n
      i = 0

    for c, freq in count.items():
      for _ in range(freq):
        ans[i] = c
        i += 2
        if i >= n:
          i = 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/recover-binary-search-tree/description/
class Solution:
  def recoverTree(self, root: Optional[TreeNode]) -> None:
    def swap(x: Optional[TreeNode], y: Optional[TreeNode]) -> None:
      temp = x.val
      x.val = y.val
      y.val = temp

    def inorder(root: Optional[TreeNode]) -> None:
      if not root:
        return

      inorder(root.left)

      if self.pred and root.val < self.pred.val:
        self.y = root
        if not self.x:
          self.x = self.pred
        else:
          return
      self.pred = root

      inorder(root.right)

    inorder(root)
    swap(self.x, self.y)

  pred = None
  x = None  # the first wrong node
  y = None  # the second wrong node


# Link: https://leetcode.com/problems/recover-binary-search-tree/description/
class Solution:
  def recoverTree(self, root: Optional[TreeNode]) -> None:
    pred = None
    x = None  # the first wrong node
    y = None  # the second wrong node

    def findPredecessor(root: Optional[TreeNode]) -> Optional[TreeNode]:
      pred = root.left
      while pred.right and pred.right != root:
        pred = pred.right
      return pred

    while root:
      if root.left:
        morrisPred = findPredecessor(root)
        if morrisPred.right:
          # The node has already been connected before.
          # Start the main logic.
          if pred and root.val < pred.val:
            y = root
            if not x:
              x = pred
          pred = root
          # End of the main logic
          morrisPred.right = None  # Break the connection.
          root = root.right
        else:
          morrisPred.right = root  # Connect it.
          root = root.left
      else:
        # Start the main logic.
        if pred and root.val < pred.val:
          y = root
          if not x:
            x = pred
        pred = root
        # End of the main logic.
        root = root.right

    def swap(x: Optional[TreeNode], y: Optional[TreeNode]) -> None:
      temp = x.val
      x.val = y.val
      y.val = temp

    swap(x, y)


# Link: https://leetcode.com/problems/recover-binary-search-tree/description/
class Solution:
  def recoverTree(self, root: Optional[TreeNode]) -> None:
    pred = None
    x = None  # the first wrong node
    y = None  # the second wrong node
    stack = []

    while root or stack:
      while root:
        stack.append(root)
        root = root.left
      root = stack.pop()
      if pred and root.val < pred.val:
        y = root
        if not x:
          x = pred
      pred = root
      root = root.right

    def swap(x: Optional[TreeNode], y: Optional[TreeNode]) -> None:
      temp = x.val
      x.val = y.val
      y.val = temp

    swap(x, y)


# Link: https://leetcode.com/problems/remove-comments/description/
class Solution:
  def removeComments(self, source: List[str]) -> List[str]:
    ans = []
    commenting = False
    modified = ''

    for line in source:
      i = 0
      while i < len(line):
        if i + 1 == len(line):
          if not commenting:
            modified += line[i]
          i += 1
          break
        twoChars = line[i:i + 2]
        if twoChars == '/*' and not commenting:
          commenting = True
          i += 2
        elif twoChars == '*/' and commenting:
          commenting = False
          i += 2
        elif twoChars == '//':
          if not commenting:
            break
          else:
            i += 2
        else:
          if not commenting:
            modified += line[i]
          i += 1
      if modified and not commenting:
        ans.append(modified)
        modified = ''

    return ans


# Link: https://leetcode.com/problems/reverse-linked-list-ii/description/
class Solution:
  def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    if left == 1:
      return self.reverseN(head, right)

    head.next = self.reverseBetween(head.next, left - 1, right - 1)
    return head

  def reverseN(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    if n == 1:
      return head

    newHead = self.reverseN(head.next, n - 1)
    headNext = head.next
    head.next = headNext.next
    headNext.next = head
    return newHead


# Link: https://leetcode.com/problems/reverse-linked-list-ii/description/
class Solution:
  def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
    if not head and m == n:
      return head

    dummy = ListNode(0, head)
    prev = dummy

    for _ in range(m - 1):
      prev = prev.next  # Point to the node before the sublist [m, n].

    tail = prev.next  # Be the tail of the sublist [m, n].

    # Reverse the sublist [m, n] one by one.
    for _ in range(n - m):
      cache = tail.next
      tail.next = cache.next
      cache.next = prev.next
      prev.next = cache

    return dummy.next


# Link: https://leetcode.com/problems/replace-words/description/
class Solution:
  def __init__(self):
    self.root = {}

  def insert(self, word: str) -> None:
    node = self.root
    for c in word:
      if c not in node:
        node[c] = {}
      node = node[c]
    node['word'] = word

  def search(self, word: str) -> str:
    node = self.root
    for c in word:
      if 'word' in node:
        return node['word']
      if c not in node:
        return word
      node = node[c]
    return word

  def replaceWords(self, dictionary: List[str], sentence: str) -> str:
    for word in dictionary:
      self.insert(word)

    words = sentence.split(' ')
    return ' '.join([self.search(word) for word in words])


# Link: https://leetcode.com/problems/broken-calculator/description/
class Solution:
  def brokenCalc(self, X: int, Y: int) -> int:
    ops = 0

    while X < Y:
      if Y % 2 == 0:
        Y //= 2
      else:
        Y += 1
      ops += 1

    return ops + X - Y


# Link: https://leetcode.com/problems/find-the-maximum-number-of-marked-indices/description/
class Solution:
  def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
    nums.sort()

    def isPossible(m: int) -> bool:
      for i in range(m):
        if 2 * nums[i] > nums[-m + i]:
          return False
      return True

    l = bisect.bisect_left(range(len(nums) // 2 + 1), True,
                           key=lambda m: not isPossible(m))
    return (l - 1) * 2


# Link: https://leetcode.com/problems/find-the-maximum-number-of-marked-indices/description/
class Solution:
  def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
    nums.sort()

    i = 0
    for j in range(len(nums) // 2, len(nums)):
      if 2 * nums[i] <= nums[j]:
        i += 1
        if i == len(nums) // 2:
          break

    return i * 2


# Link: https://leetcode.com/problems/binary-subarrays-with-sum/description/
class Solution:
  def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
    def numSubarraysWithSumAtMost(goal: int) -> int:
      res = 0
      count = 0
      l = 0
      r = 0

      while r < len(nums):
        count += nums[r]
        r += 1
        while l < r and count > goal:
          count -= nums[l]
          l += 1
        # nums[l..r), nums[l + 1..r), ..., nums[r - 1]
        res += r - l

      return res

    return numSubarraysWithSumAtMost(goal) - numSubarraysWithSumAtMost(goal - 1)


# Link: https://leetcode.com/problems/binary-subarrays-with-sum/description/
class Solution:
  def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
    ans = 0
    prefix = 0
    # {prefix: number of occurrence}
    count = collections.Counter({0: 1})

    for num in nums:
      prefix += num
      ans += count[prefix - goal]
      count[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/number-of-divisible-triplet-sums/description/
class Solution:
  # Similar to 1995. Count Special Quadruplets
  def divisibleTripletCount(self, nums: List[int], d: int) -> int:
    ans = 0
    count = collections.Counter()

    for j in range(len(nums) - 1, 0, -1):  # `j` also represents k.
      for i in range(j - 1, -1, -1):
        ans += count[-(nums[i] + nums[j]) % d]
      count[nums[j] % d] += 1  # j := k

    return ans


# Link: https://leetcode.com/problems/delete-node-in-a-bst/description/
class Solution:
  def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    if not root:
      return None
    if root.val == key:
      if not root.left:
        return root.right
      if not root.right:
        return root.left
      minNode = self._getMin(root.right)
      root.right = self.deleteNode(root.right, minNode.val)
      minNode.left = root.left
      minNode.right = root.right
      root = minNode
    elif root.val < key:
      root.right = self.deleteNode(root.right, key)
    else:  # root.val > key
      root.left = self.deleteNode(root.left, key)
    return root

  def _getMin(self, node: Optional[TreeNode]) -> Optional[TreeNode]:
    while node.left:
      node = node.left
    return node


# Link: https://leetcode.com/problems/smallest-number-with-given-digit-product/description/
class Solution:
  def smallestNumber(self, n: int) -> str:
    if n <= 9:
      return str(n)

    ans = []

    for divisor in range(9, 1, -1):
      while n % divisor == 0:
        ans.append(str(divisor))
        n //= divisor

    return '-1' if n > 1 else ''.join(reversed(ans))


# Link: https://leetcode.com/problems/neighboring-bitwise-xor/description/
class Solution:
  def doesValidArrayExist(self, derived: List[int]) -> bool:
    return functools.reduce(operator.xor, derived) == 0


# Link: https://leetcode.com/problems/filling-bookcase-shelves/description/
class Solution:
  def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
    # dp[i] := the minimum height to place the first i books
    dp = [0] + [math.inf] * len(books)

    for i in range(len(books)):
      sumThickness = 0
      maxHeight = 0
      # Place books[j..i] on a new shelf.
      for j in range(i, -1, -1):
        thickness, height = books[j]
        sumThickness += thickness
        if sumThickness > shelfWidth:
          break
        maxHeight = max(maxHeight, height)
        dp[i + 1] = min(dp[i + 1], dp[j] + maxHeight)

    return dp[-1]


# Link: https://leetcode.com/problems/maximum-product-subarray/description/
class Solution:
  def maxProduct(self, nums: List[int]) -> int:
    ans = nums[0]
    dpMin = nums[0]  # the minimum so far
    dpMax = nums[0]  # the maximum so far

    for i in range(1, len(nums)):
      num = nums[i]
      prevMin = dpMin  # dpMin[i - 1]
      prevMax = dpMax  # dpMax[i - 1]
      if num < 0:
        dpMin = min(prevMax * num, num)
        dpMax = max(prevMin * num, num)
      else:
        dpMin = min(prevMin * num, num)
        dpMax = max(prevMax * num, num)

      ans = max(ans, dpMax)

    return ans


# Link: https://leetcode.com/problems/find-all-good-indices/description/
class Solution:
  # Same as 2100. Find Good Days to Rob the Bank
  def goodIndices(self, nums: List[int], k: int) -> List[int]:
    n = len(nums)
    dec = [1] * n  # 1 + the number of continuous decreasing numbers before i
    inc = [1] * n  # 1 + the number of continuous increasing numbers after i

    for i in range(1, n):
      if nums[i - 1] >= nums[i]:
        dec[i] = dec[i - 1] + 1

    for i in range(n - 2, -1, -1):
      if nums[i] <= nums[i + 1]:
        inc[i] = inc[i + 1] + 1

    return [i for i in range(k, n - k)
            if dec[i - 1] >= k and inc[i + 1] >= k]


# Link: https://leetcode.com/problems/largest-number-after-mutating-substring/description/
class Solution:
  def maximumNumber(self, num: str, change: List[int]) -> str:
    numList = list(num)
    mutated = False

    for i, c in enumerate(numList):
      d = int(c)
      numList[i] = chr(ord('0') + max(d, change[d]))
      if mutated and d > change[d]:
        return ''.join(numList)
      if d < change[d]:
        mutated = True

    return ''.join(numList)


# Link: https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/description/
class Solution:
  def getHappyString(self, n: int, k: int) -> str:
    nextLetters = {'a': 'bc', 'b': 'ac', 'c': 'ab'}
    q = collections.deque(['a', 'b', 'c'])

    while len(q[0]) != n:
      u = q.popleft()
      for nextLetter in nextLetters[u[-1]]:
        q.append(u + nextLetter)

    return '' if len(q) < k else q[k - 1]


# Link: https://leetcode.com/problems/minimum-path-sum/description/
class Solution:
  def minPathSum(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    for i in range(m):
      for j in range(n):
        if i > 0 and j > 0:
          grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        elif i > 0:
          grid[i][0] += grid[i - 1][0]
        elif j > 0:
          grid[0][j] += grid[0][j - 1]

    return grid[m - 1][n - 1]


# Link: https://leetcode.com/problems/maximum-alternating-subarray-sum/description/
class Solution:
  def maximumAlternatingSubarraySum(self, nums: List[int]) -> int:
    ans = -math.inf
    even = 0  # the subarray sum starting from an even index
    odd = 0  # the subarray sum starting from an odd index

    for i in range(len(nums)):
      if (i & 1) == 0:  # Must pick.
        even += nums[i]
      else:  # Start a fresh subarray or subtract `nums[i]`.
        even = max(0, even - nums[i])
      ans = max(ans, even)

    for i in range(1, len(nums)):
      if i & 1:  # Must pick.
        odd += nums[i]
      else:  # Start a fresh subarray or subtract `nums[i]`.
        odd = max(0, odd - nums[i])
      ans = max(ans, odd)

    return ans


# Link: https://leetcode.com/problems/merge-operations-to-turn-array-into-a-palindrome/description/
class Solution:
  def minimumOperations(self, nums: List[int]) -> int:
    ans = 0
    l = 0
    r = len(nums) - 1
    leftSum = nums[0]
    rightSum = nums[-1]

    while l < r:
      if leftSum < rightSum:
        l += 1
        leftSum += nums[l]
        ans += 1
      elif leftSum > rightSum:
        r -= 1
        rightSum += nums[r]
        ans += 1
      else:  # leftSum == rightSum
        l += 1
        r -= 1
        leftSum = nums[l]
        rightSum = nums[r]

    return ans


# Link: https://leetcode.com/problems/ways-to-express-an-integer-as-sum-of-powers/description/
class Solution:
  def numberOfWays(self, n: int, x: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of ways to express i
    dp = [1] + [0] * n

    for a in range(1, n + 1):
      ax = a**x
      if ax > n:
        break
      for i in range(n, ax - 1, -1):
        dp[i] += dp[i - ax]
        dp[i] %= kMod

    return dp[n]


# Link: https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/description/
class Solution:
  def removeZeroSumSublists(self, head: ListNode) -> ListNode:
    dummy = ListNode(0, head)
    prefix = 0
    prefixToNode = {0: dummy}

    while head:
      prefix += head.val
      prefixToNode[prefix] = head
      head = head.next

    prefix = 0
    head = dummy

    while head:
      prefix += head.val
      head.next = prefixToNode[prefix].next
      head = head.next

    return dummy.next


# Link: https://leetcode.com/problems/binary-tree-pruning/description/
class Solution:
  def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
      return None
    root.left = self.pruneTree(root.left)
    root.right = self.pruneTree(root.right)
    if not root.left and not root.right and not root.val:
      return None
    return root


# Link: https://leetcode.com/problems/camelcase-matching/description/
class Solution:
  def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
    def isMatch(query: str) -> bool:
      j = 0
      for c in query:
        if j < len(pattern) and c == pattern[j]:
          j += 1
        elif c.isupper():
          return False
      return j == len(pattern)

    return [isMatch(query) for query in queries]


# Link: https://leetcode.com/problems/clumsy-factorial/description/
class Solution:
  def clumsy(self, n: int) -> int:
    if n == 1:
      return 1
    if n == 2:
      return 2
    if n == 3:
      return 6
    if n == 4:
      return 7
    if n % 4 == 1:
      return n + 2
    if n % 4 == 2:
      return n + 2
    if n % 4 == 3:
      return n - 1
    return n + 1


# Link: https://leetcode.com/problems/earliest-second-to-mark-indices-i/description/
class Solution:
  def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
    def canMark(second: int) -> bool:
      """
      Returns True if all indices of `nums` can be marked within `second`.
      """
      numMarked = 0
      decrement = 0
      indexToLastSecond = {}

      for i in range(second):
        indexToLastSecond[changeIndices[i] - 1] = i

      for i in range(second):
        index = changeIndices[i] - 1  # Convert to 0-indexed
        if i == indexToLastSecond[index]:
          # Reach the last occurrence of the number.
          # So, the current second will be used to mark the index.
          if nums[index] > decrement:
            # The decrement is less than the number to be marked.
            return False
          decrement -= nums[index]
          numMarked += 1
        else:
          decrement += 1

      return numMarked == len(nums)

    l = bisect.bisect_left(range(1, len(changeIndices) + 1), True,
                           key=lambda m: canMark(m)) + 1
    return l if l <= len(changeIndices) else -1


# Link: https://leetcode.com/problems/minimum-jumps-to-reach-home/description/
from enum import Enum


class Direction(Enum):
  kForward = 0
  kBackward = 1


class Solution:
  def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
    furthest = max(x + a + b, max(pos + a + b for pos in forbidden))
    seenForward = {pos for pos in forbidden}
    seenBackward = {pos for pos in forbidden}

    # (direction, position)
    q = collections.deque([(Direction.kForward, 0)])

    ans = 0
    while q:
      for _ in range(len(q)):
        dir, pos = q.popleft()
        if pos == x:
          return ans
        forward = pos + a
        backward = pos - b
        if forward <= furthest and forward not in seenForward:
          seenForward.add(forward)
          q.append((Direction.kForward, forward))
        # It cannot jump backward twice in a row.
        if dir == Direction.kForward and backward >= 0 and backward not in seenBackward:
          seenBackward.add(backward)
          q.append((Direction.kBackward, backward))
      ans += 1

    return -1


# Link: https://leetcode.com/problems/previous-permutation-with-one-swap/description/
class Solution:
  def prevPermOpt1(self, arr: List[int]) -> List[int]:
    n = len(arr)
    l = n - 2
    r = n - 1

    while l >= 0 and arr[l] <= arr[l + 1]:
      l -= 1
    if l < 0:
      return arr
    while arr[r] >= arr[l] or arr[r] == arr[r - 1]:
      r -= 1
    arr[l], arr[r] = arr[r], arr[l]

    return arr


# Link: https://leetcode.com/problems/gas-station/description/
class Solution:
  def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    ans = 0
    net = 0
    summ = 0

    # Try to start from each index.
    for i in range(len(gas)):
      net += gas[i] - cost[i]
      summ += gas[i] - cost[i]
      if summ < 0:
        summ = 0
        ans = i + 1  # Start from the next index.

    return -1 if net < 0 else ans


# Link: https://leetcode.com/problems/the-number-of-the-smallest-unoccupied-chair/description/
class Solution:
  def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
    nextUnsatChair = 0
    emptyChairs = []
    occupied = []  # (leaving, chair)

    for i in range(len(times)):
      times[i].append(i)

    times.sort(key=lambda time: time[0])

    for arrival, leaving, i in times:
      while len(occupied) > 0 and occupied[0][0] <= arrival:
        unsatChair = heapq.heappop(occupied)[1]
        heapq.heappush(emptyChairs, unsatChair)
      if i == targetFriend:
        return emptyChairs[0] if len(emptyChairs) > 0 else nextUnsatChair
      if len(emptyChairs) == 0:
        heapq.heappush(occupied, (leaving, nextUnsatChair))
        nextUnsatChair += 1
      else:
        emptyChair = heapq.heappop(emptyChairs)
        heapq.heappush(occupied, (leaving, emptyChair))


# Link: https://leetcode.com/problems/ip-to-cidr/description/
class Solution:
  def ipToCIDR(self, ip: str, n: int) -> List[str]:
    ans = []
    num = self._getNum(ip.split('.'))

    while n > 0:
      lowbit = num & -num
      count = self._maxLow(n) if lowbit == 0 else self._firstFit(lowbit, n)
      ans.append(self._getCIDR(num, self._getPrefix(count)))
      n -= count
      num += count

    return ans

  def _getNum(self, x: List[str]) -> int:
    num = 0
    for i in range(4):
      num = num * 256 + int(x[i])
    return num

  def _maxLow(self, n: int) -> Optional[int]:
    """Returns the maximum i s.t. 2^i < n."""
    for i in range(32):
      if 1 << i + 1 > n:
        return 1 << i

  def _firstFit(self, lowbit: int, n: int) -> int:
    while lowbit > n:
      lowbit >>= 1
    return lowbit

  def _getCIDR(self, num: int, prefix: int) -> str:
    d = num & 255
    num >>= 8
    c = num & 255
    num >>= 8
    b = num & 255
    num >>= 8
    a = num & 255
    return '.'.join([str(s) for s in [a, b, c, d]]) + '/' + str(prefix)

  def _getPrefix(self, count: int) -> Optional[int]:
    """
    e.g. count = 8 = 2^3 . prefix = 32 - 3 = 29
         count = 1 = 2^0 . prefix = 32 - 0 = 32
    """
    for i in range(32):
      if count == 1 << i:
        return 32 - i


# Link: https://leetcode.com/problems/flower-planting-with-no-adjacent/description/
class Solution:
  def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
    ans = [0] * n  # ans[i] := 1, 2, 3, or 4
    graph = [[] for _ in range(n)]

    for a, b in paths:
      u = a - 1
      v = b - 1
      graph[u].append(v)
      graph[v].append(u)

    for i in range(n):
      used = [False] * 5
      for v in graph[i]:
        used[ans[v]] = True
      for type in range(1, 5):
        if not used[type]:
          ans[i] = type
          break

    return ans


# Link: https://leetcode.com/problems/task-scheduler-ii/description/
class Solution:
  def taskSchedulerII(self, tasks: List[int], space: int) -> int:
    taskToNextAvailable = collections.defaultdict(int)
    ans = 0

    for task in tasks:
      ans = max(ans + 1, taskToNextAvailable[task])
      taskToNextAvailable[task] = ans + space + 1

    return ans


# Link: https://leetcode.com/problems/find-peak-element/description/
class Solution:
  def findPeakElement(self, nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1

    while l < r:
      m = (l + r) // 2
      if nums[m] >= nums[m + 1]:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/generate-random-point-in-a-circle/description/
class Solution:
  def __init__(self, radius: float, x_center: float, y_center: float):
    self.radius = radius
    self.x_center = x_center
    self.y_center = y_center

  def randPoint(self) -> List[float]:
    length = sqrt(random.uniform(0, 1)) * self.radius
    degree = random.uniform(0, 1) * 2 * math.pi
    x = self.x_center + length * math.cos(degree)
    y = self.y_center + length * math.sin(degree)
    return [x, y]


# Link: https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/description/
class Solution:
  def maxDepthAfterSplit(self, seq: str) -> List[int]:
    ans = []
    depth = 1

    # Put all odd-depth parentheses in one group and even-depth parentheses in the other group.
    for c in seq:
      if c == '(':
        depth += 1
        ans.append(depth % 2)
      else:
        ans.append(depth % 2)
        depth -= 1

    return ans


# Link: https://leetcode.com/problems/maximum-binary-tree/description/
class Solution:
  def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
    def build(i: int, j: int) -> Optional[TreeNode]:
      if i > j:
        return None

      maxNum = max(nums[i:j + 1])
      maxIndex = nums.index(maxNum)

      root = TreeNode(maxNum)
      root.left = build(i, maxIndex - 1)
      root.right = build(maxIndex + 1, j)
      return root

    return build(0, len(nums) - 1)


# Link: https://leetcode.com/problems/sender-with-largest-word-count/description/
class Solution:
  def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
    n = len(messages)
    ans = ''
    maxWordsSent = 0
    count = collections.Counter()  # [sender, # Words sent]

    for message, sender in zip(messages, senders):
      wordsCount = message.count(' ') + 1
      count[sender] += wordsCount
      numWordsSent = count[sender]
      if numWordsSent > maxWordsSent:
        ans = sender
        maxWordsSent = numWordsSent
      elif numWordsSent == maxWordsSent and sender > ans:
        ans = sender

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-groups-entering-a-competition/description/
class Solution:
  def maximumGroups(self, grades: List[int]) -> int:
    # Sort grades, then we can seperate the students into groups of sizes 1, 2,
    # 3, ..., k, s.t. the i-th group < the (i + 1)-th group for both sum and
    # size. So, we can rephrase the problem into:
    #   Find the maximum k s.t. 1 + 2 + 3 + ... + k <= n

    #  1 + 2 + 3 + ... + k <= n
    #         k(k + 1) // 2 <= n
    #              k^2 + k <= 2n
    #   (k + 0.5)^2 - 0.25 <= 2n
    #          (k + 0.5)^2 <= 2n + 0.25
    #                    k <= sqrt(2n + 0.25) - 0.5
    return int(math.sqrt(len(grades) * 2 + 0.25) - 0.5)


# Link: https://leetcode.com/problems/minimum-cost-to-convert-string-i/description/
class Solution:
  def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
    ans = 0
    # dist[u][v] := the minimum distance to change ('a' + u) to ('a' + v)
    dist = [[math.inf] * 26 for _ in range(26)]

    for a, b, c in zip(original, changed, cost):
      u = ord(a) - ord('a')
      v = ord(b) - ord('a')
      dist[u][v] = min(dist[u][v], c)

    for k in range(26):
      for i in range(26):
        if dist[i][k] < math.inf:
          for j in range(26):
            if dist[k][j] < math.inf:
              dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    for s, t in zip(source, target):
      if s == t:
        continue
      u = ord(s) - ord('a')
      v = ord(t) - ord('a')
      if dist[u][v] == math.inf:
        return -1
      ans += dist[u][v]

    return ans


# Link: https://leetcode.com/problems/design-bitset/description/
class Bitset:
  def __init__(self, size: int):
    self.s = ['0'] * size  # the original
    self.r = ['1'] * size  # the reversed
    self.cnt = 0

  def fix(self, idx: int) -> None:
    if self.s[idx] == '0':
      self.cnt += 1
    self.s[idx] = '1'
    self.r[idx] = '0'

  def unfix(self, idx: int) -> None:
    if self.s[idx] == '1':
      self.cnt -= 1
    self.s[idx] = '0'
    self.r[idx] = '1'

  def flip(self) -> None:
    self.s, self.r = self.r, self.s
    self.cnt = len(self.s) - self.cnt

  def all(self) -> bool:
    return self.cnt == len(self.s)

  def one(self) -> bool:
    return self.cnt

  def count(self) -> int:
    return self.cnt

  def toString(self) -> str:
    return ''.join(self.s)


# Link: https://leetcode.com/problems/find-the-largest-area-of-square-inside-two-rectangles/description/
class Solution:
  def largestSquareArea(self, bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
    minSide = 0

    for ((ax1, ay1), (ax2, ay2)), ((bx1, by1), (bx2, by2)) in itertools.combinations(zip(bottomLeft, topRight), 2):
      overlapX = min(ax2, bx2) - max(ax1, bx1)
      overlapY = min(ay2, by2) - max(ay1, by1)
      minSide = max(minSide, min(overlapX, overlapY))

    return minSide**2


# Link: https://leetcode.com/problems/masking-personal-information/description/
class Solution:
  def maskPII(self, s: str) -> str:
    atIndex = s.find('@')
    if atIndex != -1:
      s = s.lower()
      return s[0] + '*' * 5 + s[atIndex - 1:]

    ans = ''.join(c for c in s if c.isdigit())

    if len(ans) == 10:
      return '***-***-' + ans[-4:]
    return '+' + '*' * (len(ans) - 10) + '-***-***-' + ans[-4:]


# Link: https://leetcode.com/problems/k-diff-pairs-in-an-array/description/
class Solution:
  def findPairs(self, nums: List[int], k: int) -> int:
    ans = 0
    numToIndex = {num: i for i, num in enumerate(nums)}

    for i, num in enumerate(nums):
      target = num + k
      if target in numToIndex and numToIndex[target] != i:
        ans += 1
        del numToIndex[target]

    return ans


# Link: https://leetcode.com/problems/cutting-ribbons/description/
class Solution:
  def maxLength(self, ribbons: List[int], k: int) -> int:
    def isCutPossible(length: int) -> bool:
      count = 0
      for ribbon in ribbons:
        count += ribbon // length
      return count >= k

    l = 1
    r = sum(ribbons) // k + 1

    while l < r:
      m = (l + r) // 2
      if not isCutPossible(m):
        r = m
      else:
        l = m + 1

    return l - 1


# Link: https://leetcode.com/problems/watering-plants-ii/description/
class Solution:
  def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
    ans = 0
    i = 0
    j = len(plants) - 1
    canA = capacityA
    canB = capacityB

    while i < j:
      ans += (canA < plants[i]) + (canB < plants[j])
      if canA < plants[i]:
        canA = capacityA
      if canB < plants[j]:
        canB = capacityB
      canA -= plants[i]
      canB -= plants[j]
      i += 1
      j -= 1

    return ans + (i == j and max(canA, canB) < plants[i])


# Link: https://leetcode.com/problems/number-of-laser-beams-in-a-bank/description/
class Solution:
  def numberOfBeams(self, bank: List[str]) -> int:
    ans = 0
    prevOnes = 0

    for row in bank:
      ones = row.count('1')
      if ones:
        ans += prevOnes * ones
        prevOnes = ones

    return ans


# Link: https://leetcode.com/problems/brick-wall/description/
class Solution:
  def leastBricks(self, wall: List[List[int]]) -> int:
    maxFreq = 0
    count = collections.defaultdict(int)

    for row in wall:
      prefix = 0
      for i in range(len(row) - 1):
        prefix += row[i]
        count[prefix] += 1
        maxFreq = max(maxFreq, count[prefix])

    return len(wall) - maxFreq


# Link: https://leetcode.com/problems/arithmetic-slices/description/
class Solution:
  def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    ans = 0
    dp = 0

    for i in range(2, len(nums)):
      if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
        dp += 1
        ans += dp
      else:
        dp = 0

    return ans


# Link: https://leetcode.com/problems/arithmetic-slices/description/
class Solution:
  def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    n = len(nums)
    if n < 3:
      return 0

    dp = [0] * n  # dp[i] := the number of arithmetic slices ending in index i

    for i in range(2, len(nums)):
      if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
        dp[i] = dp[i - 1] + 1

    return sum(dp)


# Link: https://leetcode.com/problems/minimize-rounding-error-to-meet-target/description/
class Solution:
  def minimizeError(self, prices: List[str], target: int) -> str:
    # A[i] := (costCeil - costFloor, costCeil, costFloor)
    # The lower the costCeil - costFloor is, the cheaper to ceil it.
    A = []
    sumFloored = 0
    sumCeiled = 0

    for price in map(float, prices):
      floored = math.floor(price)
      ceiled = math.ceil(price)
      sumFloored += floored
      sumCeiled += ceiled
      costFloor = price - floored
      costCeil = ceiled - price
      A.append((costCeil - costFloor, costCeil, costFloor))

    if not sumFloored <= target <= sumCeiled:
      return '-1'

    A.sort()
    nCeiled = target - sumFloored
    return '{:.3f}'.format(sum(a[1] for a in A[:nCeiled]) +
                           sum(a[2] for a in A[nCeiled:]))


# Link: https://leetcode.com/problems/maximum-distance-between-a-pair-of-values/description/
class Solution:
  def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
    ans = 0
    i = 0
    j = 0

    while i < len(nums1) and j < len(nums2):
      if nums1[i] > nums2[j]:
        i += 1
      else:
        ans = max(ans, j - i)
        j += 1

    return ans


# Link: https://leetcode.com/problems/maximum-distance-between-a-pair-of-values/description/
class Solution:
  def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
    i = 0
    j = 0

    while i < len(nums1) and j < len(nums2):
      if nums1[i] > nums2[j]:
        i += 1
      j += 1

    return 0 if i == j else j - i - 1


# Link: https://leetcode.com/problems/count-vowel-strings-in-ranges/description/
class Solution:
  def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
    kVowels = 'aeiou'
    # prefix[i] := the number of the first i words that start with and end in a vowel
    prefix = [0] * (len(words) + 1)

    for i, word in enumerate(words):
      prefix[i + 1] += prefix[i] + (word[0] in kVowels and word[-1] in kVowels)

    return [prefix[r + 1] - prefix[l]
            for l, r in queries]


# Link: https://leetcode.com/problems/maximize-the-topmost-element-after-k-moves/description/
class Solution:
  def maximumTop(self, nums: List[int], k: int) -> int:
    n = len(nums)
    # After taking k elements, if we're left something, then we return nums[k]
    # Otherwise, return -1.
    if k == 0 or k == 1:
      return -1 if n == k else nums[k]
    # Remove then add even number of times.
    if n == 1:
      return -1 if k & 1 else nums[0]
    # Take min(n, k - 1) elements and put the largest one back.
    maxi = max(nums[:min(n, k - 1)])
    if k >= n:
      return maxi
    return max(maxi, nums[k])


# Link: https://leetcode.com/problems/odd-even-linked-list/description/
class Solution:
  def oddEvenList(self, head: ListNode) -> ListNode:
    oddHead = ListNode(0)
    evenHead = ListNode(0)
    odd = oddHead
    even = evenHead
    isOdd = True

    while head:
      if isOdd:
        odd.next = head
        odd = head
      else:
        even.next = head
        even = head
      head = head.next
      isOdd = not isOdd

    even.next = None
    odd.next = evenHead.next
    return oddHead.next


# Link: https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/description/
class Solution:
  def minSwaps(self, s: str) -> int:
    ones = s.count('1')
    zeros = len(s) - ones
    if abs(ones - zeros) > 1:
      return -1

    def countSwaps(curr: str) -> int:
      swaps = 0
      for c in s:
        if c != curr:
          swaps += 1
        curr = chr(ord(curr) ^ 1)
      return swaps // 2

    if ones > zeros:
      return countSwaps('1')
    if zeros > ones:
      return countSwaps('0')
    return min(countSwaps('1'), countSwaps('0'))


# Link: https://leetcode.com/problems/non-overlapping-intervals/description/
class Solution:
  def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    ans = 0
    currentEnd = -math.inf

    for interval in sorted(intervals, key=lambda x: x[1]):
      if interval[0] >= currentEnd:
        currentEnd = interval[1]
      else:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/filter-restaurants-by-vegan-friendly-price-and-distance/description/
class Solution:
  def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
    restaurants.sort(key=lambda r: (-r[1], -r[0]))
    return [i for i, _, v, p, d in restaurants if v >= veganFriendly and p <= maxPrice and d <= maxDistance]


# Link: https://leetcode.com/problems/print-words-vertically/description/
class Solution:
  def printVertically(self, s: str) -> List[str]:
    ans = []
    words = s.split()
    maxLength = max(len(word) for word in words)

    for i in range(maxLength):
      row = []
      for word in words:
        row.append(word[i] if i < len(word) else ' ')
      ans.append(''.join(row).rstrip())

    return ans


# Link: https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/description/
class Solution:
  def minDeletions(self, s: str) -> int:
    ans = 0
    count = collections.Counter(s)
    usedFreq = set()

    for freq in count.values():
      while freq > 0 and freq in usedFreq:
        freq -= 1  # Delete ('a' + i).
        ans += 1
      usedFreq.add(freq)

    return ans


# Link: https://leetcode.com/problems/remove-all-occurrences-of-a-substring/description/
class Solution:
  def removeOccurrences(self, s: str, part: str) -> str:
    n = len(s)
    k = len(part)

    t = [' '] * n
    j = 0  # t's index

    for i, c in enumerate(s):
      t[j] = c
      j += 1
      if j >= k and ''.join(t[j - k:j]) == part:
        j -= k

    return ''.join(t[:j])


# Link: https://leetcode.com/problems/spiral-matrix-iv/description/
class Solution:
  def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    ans = [[-1] * n for _ in range(m)]
    x = 0  # the current x position
    y = 0  # the current y position
    d = 0

    curr = head
    while curr:
      ans[x][y] = curr.val
      if x + dirs[d] < 0 or x + dirs[d] == m or y + dirs[d + 1] < 0 or \
              y + dirs[d + 1] == n or ans[x + dirs[d]][y + dirs[d + 1]] != -1:
        d = (d + 1) % 4
      x += dirs[d]
      y += dirs[d + 1]
      curr = curr.next

    return ans


# Link: https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/description/
class Solution:
  def longestSubsequence(self, arr: List[int], difference: int) -> int:
    ans = 0
    lengthAt = {}

    for a in arr:
      lengthAt[a] = lengthAt.get(a - difference, 0) + 1
      ans = max(ans, lengthAt[a])

    return ans


# Link: https://leetcode.com/problems/permutations/description/
class Solution:
  def permute(self, nums: List[int]) -> List[List[int]]:
    ans = []
    used = [False] * len(nums)

    def dfs(path: List[int]) -> None:
      if len(path) == len(nums):
        ans.append(path.copy())
        return

      for i, num in enumerate(nums):
        if used[i]:
          continue
        used[i] = True
        path.append(num)
        dfs(path)
        path.pop()
        used[i] = False

    dfs([])
    return ans


# Link: https://leetcode.com/problems/sell-diminishing-valued-colored-balls/description/
class Solution:
  def maxProfit(self, inventory: List[int], orders: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    largestCount = 1

    def trapezoid(a: int, b: int) -> int:
      return (a + b) * (a - b + 1) // 2

    for a, b in itertools.pairwise(sorted(inventory, reverse=True) + [0]):
      if a > b:
        # If we are at the last inventory, or inventory[i] > inventory[i + 1].
        # In either case, we will pick inventory[i - largestCount + 1..i].
        pick = a - b
        # We have run out of orders, so we need to recalculate the number of
        # balls that we actually pick for inventory[i - largestCount + 1..i].
        if largestCount * pick >= orders:
          actualPick, remaining = divmod(orders, largestCount)
          return (ans +
                  largestCount * trapezoid(a, a - actualPick + 1) +
                  remaining * (a - actualPick)) % kMod
        ans += largestCount * trapezoid(a, a - pick + 1)
        ans %= kMod
        orders -= largestCount * pick
      largestCount += 1


# Link: https://leetcode.com/problems/find-the-value-of-the-partition/description/
class Solution:
  def findValueOfPartition(self, nums: List[int]) -> int:
    return min(b - a for a, b in itertools.pairwise(sorted(nums)))


# Link: https://leetcode.com/problems/cheapest-flights-within-k-stops/description/
class Solution:
  def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in flights:
      graph[u].append((v, w))

    return self._dijkstra(graph, src, dst, k)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, dst: int, k: int) -> int:
    dist = [[math.inf for _ in range(k + 2)] for _ in range(len(graph))]

    dist[src][k + 1] = 0
    minHeap = [(dist[src][k + 1], src, k + 1)]  # (d, u, stops)

    while minHeap:
      d, u, stops = heapq.heappop(minHeap)
      if u == dst:
        return d
      if stops == 0:
        continue
      for v, w in graph[u]:
        if d + w < dist[v][stops - 1]:
          dist[v][stops - 1] = d + w
          heapq.heappush(minHeap, (dist[v][stops - 1], v, stops - 1))

    return -1


# Link: https://leetcode.com/problems/smallest-string-with-swaps/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self.find(u)
    j = self.find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
    ans = ''
    uf = UnionFind(len(s))
    map = collections.defaultdict(list)

    for a, b in pairs:
      uf.unionByRank(a, b)

    for i, c in enumerate(s):
      map[uf.find(i)].append(c)

    for key in map.keys():
      map[key].sort(reverse=True)

    for i in range(len(s)):
      ans += map[uf.find(i)].pop()

    return ans


# Link: https://leetcode.com/problems/find-missing-observations/description/
class Solution:
  def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
    targetSum = (len(rolls) + n) * mean
    missingSum = targetSum - sum(rolls)
    if missingSum > n * 6 or missingSum < n:
      return []

    ans = [missingSum // n] * n
    for i in range(missingSum % n):
      ans[i] += 1

    return ans


# Link: https://leetcode.com/problems/can-convert-string-in-k-moves/description/
class Solution:
  def canConvertString(self, s: str, t: str, k: int) -> bool:
    if len(s) != len(t):
      return False

    # e.g. s = "aab", t = "bbc", so shiftCount[1] = 3
    # Case 1: a -> b, need 1 move.
    # Case 2: a -> b, need 1 + 26 moves.
    # Case 3: b -> c, need 1 + 26 * 2 moves.
    shiftCount = [0] * 26

    for a, b in zip(s, t):
      shiftCount[(ord(b) - ord(a) + 26) % 26] += 1

    for shift in range(1, 26):
      if shift + 26 * (shiftCount[shift] - 1) > k:
        return False

    return True


# Link: https://leetcode.com/problems/can-convert-string-in-k-moves/description/
class Solution:
  def canConvertString(self, s: str, t: str, k: int) -> bool:
    if len(s) != len(t):
      return False

    # e.g. s = "aab", t = "bbc", so shiftCount[1] = 3
    # Case 1: a -> b, need 1 move.
    # Case 2: a -> b, need 1 + 26 moves.
    # Case 3: b -> c, need 1 + 26 * 2 moves.
    shiftCount = [0] * 26

    for a, b in zip(s, t):
      shift = (ord(b) - ord(a) + 26) % 26
      if shift == 0:
        continue
      if shift + 26 * shiftCount[shift] > k:
        return False
      shiftCount[shift] += 1

    return True


# Link: https://leetcode.com/problems/visit-array-positions-to-maximize-score/description/
class Solution:
  def maxScore(self, nums: List[int], x: int) -> int:
    # Note that we always need to take nums[0], so the initial definition might
    # not hold true.

    # dp0 := the maximum score so far with `nums` ending in an even number
    dp0 = nums[0] - (x if nums[0] % 2 == 1 else 0)
    # dp0 := the maximum score so far with `nums` ending in an odd number
    dp1 = nums[0] - (x if nums[0] % 2 == 0 else 0)

    for i in range(1, len(nums)):
      if nums[i] % 2 == 0:
        dp0 = nums[i] + max(dp0, dp1 - x)
      else:
        dp1 = nums[i] + max(dp1, dp0 - x)

    return max(dp0, dp1)


# Link: https://leetcode.com/problems/closest-divisors/description/
class Solution:
  def closestDivisors(self, num: int) -> List[int]:
    for root in reversed(range(int(sqrt(num + 2)) + 1)):
      for cand in [num + 1, num + 2]:
        if cand % root == 0:
          return [root, cand // root]


# Link: https://leetcode.com/problems/minimum-size-subarray-sum/description/
class Solution:
  def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    ans = math.inf
    summ = 0
    j = 0

    for i, num in enumerate(nums):
      summ += num
      while summ >= s:
        ans = min(ans, i - j + 1)
        summ -= nums[j]
        j += 1

    return ans if ans != math.inf else 0


# Link: https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks/description/
class Solution:
  def minSessions(self, tasks: List[int], sessionTime: int) -> int:
    # Returns True if we can assign tasks[s..n) to `sessions`. Note that `sessions`
    # may be occupied by some tasks.
    def dfs(s: int, sessions: List[int]) -> bool:
      if s == len(tasks):
        return True

      for i, session in enumerate(sessions):
        # Can't assign the tasks[s] to this session.
        if session + tasks[s] > sessionTime:
          continue
        # Assign the tasks[s] to this session.
        sessions[i] += tasks[s]
        if dfs(s + 1, sessions):
          return True
        # Backtracking.
        sessions[i] -= tasks[s]
        # If it's the first time we assign the tasks[s] to this session, then future
        # `session`s can't satisfy either.
        if sessions[i] == 0:
          return False

      return False

    for numSessions in range(1, len(tasks) + 1):
      if dfs(0, [0] * numSessions):
        return numSessions


# Link: https://leetcode.com/problems/finding-the-number-of-visible-mountains/description/
class Solution:
  def visibleMountains(self, peaks: List[List[int]]) -> int:
    count = collections.Counter((x, y) for x, y in peaks)
    peaks = sorted([k for k, v in count.items() if v == 1])
    stack = []

    # Returns True if `peak1` is hidden by `peak2`
    def isHidden(peak1: List[int], peak2: List[int]) -> bool:
      x1, y1 = peak1
      x2, y2 = peak2
      return x1 - y1 >= x2 - y2 and x1 + y1 <= x2 + y2

    for i, peak in enumerate(peaks):
      while stack and isHidden(peaks[stack[-1]], peak):
        stack.pop()
      if stack and isHidden(peak, peaks[stack[-1]]):
        continue
      stack.append(i)

    return len(stack)


# Link: https://leetcode.com/problems/sum-of-subarray-minimums/description/
class Solution:
  def sumSubarrayMins(self, arr: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(arr)
    ans = 0
    # prevMin[i] := index k s.t. arr[k] is the previous minimum in arr[:i]
    prevMin = [-1] * n
    # nextMin[i] := index k s.t. arr[k] is the next minimum in arr[i + 1:]
    nextMin = [n] * n
    stack = []

    for i, a in enumerate(arr):
      while stack and arr[stack[-1]] > a:
        index = stack.pop()
        nextMin[index] = i
      if stack:
        prevMin[i] = stack[-1]
      stack.append(i)

    for i, a in enumerate(arr):
      ans += a * (i - prevMin[i]) * (nextMin[i] - i)
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/design-phone-directory/description/
class PhoneDirectory:
  def __init__(self, maxNumbers: int):
    # the next available numbers
    self.next = [i + 1 for i in range(maxNumbers - 1)] + [0]
    # the current possible available number
    self.number = 0

  def get(self) -> int:
    if self.next[self.number] == -1:
      return -1
    ans = self.number
    self.number = self.next[self.number]
    self.next[ans] = -1  # Mark as used.
    return ans

  def check(self, number: int) -> bool:
    return self.next[number] != -1

  def release(self, number: int) -> None:
    if self.next[number] != -1:
      return
    self.next[number] = self.number
    self.number = number


# Link: https://leetcode.com/problems/word-subsets/description/
class Solution:
  def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
    count = collections.Counter()

    for b in B:
      count = count | collections.Counter(b)

    return [a for a in A if collections.Counter(a) & count == count]


# Link: https://leetcode.com/problems/cousins-in-binary-tree-ii/description/
class Solution:
  def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    levelSums = []

    def dfs(root: Optional[TreeNode], level: int) -> None:
      if not root:
        return
      if len(levelSums) == level:
        levelSums.append(0)
      levelSums[level] += root.val
      dfs(root.left, level + 1)
      dfs(root.right, level + 1)

    def replace(root: Optional[TreeNode], level: int, curr: Optional[TreeNode]) -> Optional[TreeNode]:
      nextLevel = level + 1
      nextLevelCousinsSum = (levelSums[nextLevel] if nextLevel < len(levelSums) else 0) - \
          (root.left.val if root.left else 0) - \
          (root.right.val if root.right else 0)
      if root.left:
        curr.left = TreeNode(nextLevelCousinsSum)
        replace(root.left, level + 1, curr.left)
      if root.right:
        curr.right = TreeNode(nextLevelCousinsSum)
        replace(root.right, level + 1, curr.right)
      return curr

    dfs(root, 0)
    return replace(root, 0, TreeNode(0))


# Link: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/description/
class Solution:
  def flatten(self, head: 'Node') -> 'Node':
    def flatten(head: 'Node', rest: 'Node') -> 'Node':
      if not head:
        return rest

      head.next = flatten(head.child, flatten(head.next, rest))
      if head.next:
        head.next.prev = head
      head.child = None
      return head

    return flatten(head, None)


# Link: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/description/
class Solution:
  def flatten(self, head: 'Node') -> 'Node':
    curr = head

    while curr:
      if curr.child:
        cachedNext = curr.next
        curr.next = curr.child
        curr.child.prev = curr
        curr.child = None
        tail = curr.next
        while tail.next:
          tail = tail.next
        tail.next = cachedNext
        if cachedNext:
          cachedNext.prev = tail
      curr = curr.next

    return head


# Link: https://leetcode.com/problems/destroying-asteroids/description/
class Solution:
  def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
    for asteroid in sorted(asteroids):
      if mass >= asteroid:
        mass += asteroid
      else:
        return False
    return True


# Link: https://leetcode.com/problems/synonymous-sentences/description/
from sortedcontainers import SortedSet


class Solution:
  def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
    ans = SortedSet()
    graph = collections.defaultdict(list)
    q = collections.deque([text])

    for s, t in synonyms:
      graph[s].append(t)
      graph[t].append(s)

    while q:
      u = q.popleft()
      ans.add(u)
      words = u.split()
      for i, word in enumerate(words):
        for synonym in graph[word]:
          # Replace words[i] with its synonym.
          words[i] = synonym
          newText = ' '.join(words)
          if newText not in ans:
            q.append(newText)

    return list(ans)


# Link: https://leetcode.com/problems/longest-palindromic-subsequence/description/
class Solution:
  def longestPalindromeSubseq(self, s: str) -> int:
    n = len(s)
    # dp[i][j] := the length of LPS(s[i..j])
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
      dp[i][i] = 1

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        if s[i] == s[j]:
          dp[i][j] = 2 + dp[i + 1][j - 1]
        else:
          dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]


# Link: https://leetcode.com/problems/longest-palindromic-subsequence/description/
class Solution:
  def longestPalindromeSubseq(self, s: str) -> int:
    @functools.lru_cache(None)
    def dp(i: int, j: int) -> int:
      """Returns the length of LPS(s[i..j])."""
      if i > j:
        return 0
      if i == j:
        return 1
      if s[i] == s[j]:
        return 2 + dp(i + 1, j - 1)
      return max(dp(i + 1, j), dp(i, j - 1))

    return dp(0, len(s) - 1)


# Link: https://leetcode.com/problems/number-of-equal-numbers-blocks/description/
# Definition for BigArray.
# class BigArray:
#   def at(self, index: long) -> int:
#     pass
#   def size(self) -> long:
#     pass

class Solution(object):
  def countBlocks(self, nums: Optional['BigArray']) -> int:
    def countBlocks(l: int, r: int, leftValue: int, rightValue: int) -> int:
      """Returns the number of maximal blocks in nums[l..r]."""
      if leftValue == rightValue:
        return 1
      if l + 1 == r:
        return 2
      m = (l + r) // 2
      midValue = nums.at(m)
      return countBlocks(l, m, leftValue, midValue) + countBlocks(m, r, midValue, rightValue) - 1

    # Substract nums[m], which will be counted twice.
    return countBlocks(0, nums.size() - 1,
                       nums.at(0), nums.at(nums.size() - 1))


# Link: https://leetcode.com/problems/bulls-and-cows/description/
class Solution:
  def getHint(self, secret: str, guess: str) -> str:
    bulls = sum(map(operator.eq, secret, guess))
    bovine = sum(min(secret.count(x), guess.count(x)) for x in set(guess))
    return '%dA%dB' % (bulls, bovine - bulls)


# Link: https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/description/
class Solution:
  def minPairSum(self, nums: List[int]) -> int:
    nums.sort()
    return max(nums[i] + nums[len(nums) - 1 - i] for i in range(len(nums) // 2))


# Link: https://leetcode.com/problems/flatten-2d-vector/description/
class Vector2D:
  def __init__(self, vec: List[List[int]]):
    self.vec = []
    self.i = 0

    for A in vec:
      self.vec += A

  def next(self) -> int:
    ans = self.vec[self.i]
    self.i += 1
    return ans

  def hasNext(self) -> bool:
    return self.i < len(self.vec)


# Link: https://leetcode.com/problems/minimum-operations-to-maximize-last-elements-in-arrays/description/
class Solution:
  def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
    n = len(nums1)
    mini = min(nums1[-1], nums2[-1])
    maxi = max(nums1[-1], nums2[-1])
    # the number of the minimum operations, where nums1[n - 1] is not swapped
    # with nums2[n - 1]
    dp1 = 0
    # the number of the minimum operations, where nums1[n - 1] is swapped with
    # nums2[n - 1]
    dp2 = 0

    for a, b in zip(nums1, nums2):
      if min(a, b) > mini:
        return -1
      if max(a, b) > maxi:
        return -1
      if a > nums1[-1] or b > nums2[-1]:
        dp1 += 1
      if a > nums2[-1] or b > nums1[-1]:
        dp2 += 1

    return min(dp1, dp2)


# Link: https://leetcode.com/problems/find-distance-in-a-binary-tree/description/
class Solution:
  def findDistance(self, root: TreeNode, p: int, q: int) -> int:
    def getLCA(root, p, q):
      if not root or root.val == p or root.val == q:
        return root

      l = getLCA(root.left, p, q)
      r = getLCA(root.right, p, q)

      if l and r:
        return root
      return l or r

    def dist(lca, target):
      if not lca:
        return 10000
      if lca.val == target:
        return 0
      return 1 + min(dist(lca.left, target), dist(lca.right, target))

    lca = getLCA(root, p, q)
    return dist(lca, p) + dist(lca, q)


# Link: https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/description/
class Solution:
  def minSwaps(self, nums: List[int]) -> int:
    n = len(nums)
    k = nums.count(1)
    ones = 0  # the number of ones in the window
    maxOnes = 0  # the maximum number of ones in the window

    for i in range(n * 2):
      if i >= k and nums[i % n - k]:  # Magic in Python :)
        ones -= 1
      if nums[i % n]:
        ones += 1
      maxOnes = max(maxOnes, ones)

    return k - maxOnes


# Link: https://leetcode.com/problems/max-chunks-to-make-sorted/description/
class Solution:
  def maxChunksToSorted(self, arr: List[int]) -> int:
    ans = 0
    maxi = -math.inf

    for i, a in enumerate(arr):
      maxi = max(maxi, a)
      if maxi == i:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/top-k-frequent-words/description/
class T:
  def __init__(self, word: str, freq: int):
    self.word = word
    self.freq = freq

  def __lt__(self, other):
    if self.freq == other.freq:
      # Words with higher frequency and lower alphabetical order are in the
      # bottom of the heap because we'll pop words with lower frequency and
      # higher alphabetical order if the heap's size > k.
      return self.word > other.word
    return self.freq < other.freq


class Solution:
  def topKFrequent(self, words: List[str], k: int) -> List[str]:
    ans = []
    heap = []

    for word, freq in collections.Counter(words).items():
      heapq.heappush(heap, T(word, freq))
      if len(heap) > k:
        heapq.heappop(heap)

    while heap:
      ans.append(heapq.heappop(heap).word)

    return ans[::-1]


# Link: https://leetcode.com/problems/top-k-frequent-words/description/
class Solution:
  def topKFrequent(self, words: List[str], k: int) -> List[str]:
    ans = []
    bucket = [[] for _ in range(len(words) + 1)]

    for word, freq in collections.Counter(words).items():
      bucket[freq].append(word)

    for b in reversed(bucket):
      for word in sorted(b):
        ans.append(word)
        if len(ans) == k:
          return ans


# Link: https://leetcode.com/problems/pancake-sorting/description/
class Solution:
  def pancakeSort(self, arr: List[int]) -> List[int]:
    ans = []

    for target in range(len(arr), 0, -1):
      index = arr.index(target)
      arr[:index + 1] = arr[:index + 1][::-1]
      arr[:target] = arr[:target][::-1]
      ans.append(index + 1)
      ans.append(target)

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/
class Solution:
  def maxVowels(self, s: str, k: int) -> int:
    ans = 0
    maxi = 0
    kVowels = 'aeiou'

    for i, c in enumerate(s):
      if c in kVowels:
        maxi += 1
      if i >= k and s[i - k] in kVowels:
        maxi -= 1
      ans = max(ans, maxi)

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/description/
class Solution:
  # Same as 3014. Minimum Number of Pushes to Type Word I
  def minimumPushes(self, word: str) -> int:
    freqs = sorted(collections.Counter(word).values(), reverse=True)
    return sum(freq * (i // 8 + 1) for i, freq in enumerate(freqs))


# Link: https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/description/
class Solution:
  def partitionArray(self, nums: List[int], k: int) -> int:
    nums.sort()

    ans = 1
    min = nums[0]

    for i in range(1, len(nums)):
      if min + k < nums[i]:
        ans += 1
        min = nums[i]

    return ans


# Link: https://leetcode.com/problems/count-good-numbers/description/
class Solution:
  def countGoodNumbers(self, n: int) -> int:
    kMod = 1_000_000_007

    def modPow(x: int, n: int) -> int:
      if n == 0:
        return 1
      if n & 1:
        return x * modPow(x, n - 1) % kMod
      return modPow(x * x % kMod, n // 2)

    return modPow(4 * 5, n // 2) * (5 if n & 1 else 1) % kMod


# Link: https://leetcode.com/problems/arithmetic-subarrays/description/
class Solution:
  def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
    return [self._isArithmetic(nums, a, b) for a, b in zip(l, r)]

  def _isArithmetic(self, nums: List[int], l: int, r: int) -> bool:
    if r - l < 2:
      return True

    numsSet = set()
    mini = math.inf
    maxi = -math.inf

    for i in range(l, r+1):
      mini = min(mini, nums[i])
      maxi = max(maxi, nums[i])
      numsSet.add(nums[i])

    if (maxi - mini) % (r - l) != 0:
      return False

    interval = (maxi - mini) // (r - l)
    return all(mini + k * interval in numsSet
               for k in range(1, r - l + 1))


# Link: https://leetcode.com/problems/binary-tree-level-order-traversal/description/
class Solution:
  def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
      return []

    ans = []
    q = collections.deque([root])

    while q:
      currLevel = []
      for _ in range(len(q)):
        node = q.popleft()
        currLevel.append(node.val)
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      ans.append(currLevel)

    return ans


# Link: https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/description/
class Solution:
  def countPairs(self, n: int, edges: List[List[int]]) -> int:
    ans = 0
    graph = [0] * n
    seen = [0] * n
    unreached = n

    for e in edges:
      u = e[0]
      v = e[1]
      graph[u].append(v)
      graph[v].append(u)

    for i in range(n):
      reached = dfs(graph, i, seen)
      unreached -= reached
      ans += static_cast < long > (unreached) * reached

    return ans

  def dfs(self, graph: List[List[int]], u: int, seen: List[bool]) -> int:
    if seen[u]:
      return 0
    seen[u] = True
    return accumulate(
        begin(graph[u]), end(graph[u]), 1,
        [ & ](subtotal, v) [return subtotal + dfs(graph, v, seen)])


# Link: https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/description/
class Solution:
  def isPossibleDivide(self, nums: List[int], k: int) -> bool:
    count = collections.Counter(nums)

    for start in sorted(count):
      value = count[start]
      if value > 0:
        for i in range(start, start + k):
          count[i] -= value
          if count[i] < 0:
            return False

    return True


# Link: https://leetcode.com/problems/card-flipping-game/description/
class Solution:
  def flipgame(self, fronts: List[int], backs: List[int]) -> int:
    same = {front
            for front, back in zip(fronts, backs)
            if front == back}
    return min([num for num in fronts + backs
                if num not in same] or [0])


# Link: https://leetcode.com/problems/maximum-number-of-jumps-to-reach-the-last-index/description/
class Solution:
  def maximumJumps(self, nums: List[int], target: int) -> int:
    n = len(nums)
    # dp[i] := the maximum number of jumps to reach i from 0
    dp = [-1] * n
    dp[0] = 0

    for j in range(1, n):
      for i in range(j):
        if dp[i] != -1 and abs(nums[j] - nums[i]) <= target:
          dp[j] = max(dp[j], dp[i] + 1)

    return dp[-1]


# Link: https://leetcode.com/problems/design-log-storage-system/description/
class LogSystem:
  def __init__(self):
    self.granularityToIndices = {'Year': 4, 'Month': 7, 'Day': 10,
                                 'Hour': 13, 'Minute': 16, 'Second': 19}
    self.idAndTimestamps = []

  def put(self, id: int, timestamp: str) -> None:
    self.idAndTimestamps.append((id, timestamp))

  def retrieve(self, start: str, end: str, granularity: str) -> List[int]:
    index = self.granularityToIndices[granularity]
    s = start[:index]
    e = end[:index]
    return [id for id, timestamp in self.idAndTimestamps
            if s <= timestamp[:index] <= e]


# Link: https://leetcode.com/problems/number-of-self-divisible-permutations/description/
class Solution:
  def selfDivisiblePermutationCount(self, n: int) -> int:
    def dfs(num: int, used: int) -> int:
      if num > n:
        return 1

      count = 0
      for i in range(1, n + 1):
        if (used >> i & 1) == 0 and (num % i == 0 or i % num == 0):
          count += dfs(num + 1, used | 1 << i)

      return count

    return dfs(1, 0)


# Link: https://leetcode.com/problems/find-the-maximum-number-of-elements-in-subset/description/
class Solution:
  def maximumLength(self, nums: List[int]) -> int:
    maxNum = max(nums)
    count = collections.Counter(nums)
    ans = count[1] - (count[1] % 2 == 0) if 1 in count else 1

    for num in nums:
      if num == 1:
        continue
      length = 0
      x = num
      while x <= maxNum and x in count and count[x] >= 2:
        length += 2
        x *= x
      # x is now x^k, and the pattern is [x, ..., x^(k/2), x^(k/2), ..., x].
      # The goal is to determine if we can insert x^k in the middle of the
      # pattern to increase the length by 1. If not, we make x^(k/2) the middle
      # and decrease the length by 1.
      ans = max(ans, length + (1 if x in count else -1))

    return ans


# Link: https://leetcode.com/problems/brightest-position-on-street/description/
from sortedcontainers import SortedDict


class Solution:
  def brightestPosition(self, lights: List[List[int]]) -> int:
    ans = math.inf
    maxBrightness = -1
    currBrightness = 0
    line = SortedDict()

    for position, rg in lights:
      start = position - rg
      end = position + rg + 1
      line[start] = line.get(start, 0) + 1
      line[end] = line.get(end, 0) - 1

    for pos, brightness in line.items():
      currBrightness += brightness
      if currBrightness > maxBrightness:
        maxBrightness = currBrightness
        ans = pos

    return ans


# Link: https://leetcode.com/problems/count-of-interesting-subarrays/description/
class Solution:
  def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
    ans = 0
    prefix = 0  # (number of nums[i] % modulo == k so far) % modulo
    prefixCount = collections.Counter({0: 1})

    for num in nums:
      if num % modulo == k:
        prefix = (prefix + 1) % modulo
      ans += prefixCount[(prefix - k + modulo) % modulo]
      prefixCount[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/group-anagrams/description/
class Solution:
  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    dict = collections.defaultdict(list)

    for str in strs:
      key = ''.join(sorted(str))
      dict[key].append(str)

    return dict.values()


# Link: https://leetcode.com/problems/add-two-polynomials-represented-as-linked-lists/description/
# Definition for polynomial singly-linked list.
# class PolyNode:
#   def __init__(self, x=0, y=0, next=None):
#     self.coefficient = x
#     self.power = y
#     self.next = next

class Solution:
  def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
    dummy = PolyNode()
    curr = dummy
    p = poly1  # poly1's pointer
    q = poly2  # poly2's pointer

    while p and q:
      if p.power > q.power:
        curr.next = PolyNode(p.coefficient, p.power)
        curr = curr.next
        p = p.next
      elif p.power < q.power:
        curr.next = PolyNode(q.coefficient, q.power)
        curr = curr.next
        q = q.next
      else:  # p.power == q.power
        sumCoefficient = p.coefficient + q.coefficient
        if sumCoefficient != 0:
          curr.next = PolyNode(sumCoefficient, p.power)
          curr = curr.next
        p = p.next
        q = q.next

    while p:
      curr.next = PolyNode(p.coefficient, p.power)
      curr = curr.next
      p = p.next

    while q:
      curr.next = PolyNode(q.coefficient, q.power)
      curr = curr.next
      q = q.next

    return dummy.next


# Link: https://leetcode.com/problems/random-pick-with-weight/description/
class Solution:
  def __init__(self, w: List[int]):
    self.prefix = list(itertools.accumulate(w))

  def pickIndex(self) -> int:
    target = random.randint(0, self.prefix[-1] - 1)
    return bisect.bisect_right(range(len(self.prefix)), target,
                               key=lambda m: self.prefix[m])


# Link: https://leetcode.com/problems/random-pick-with-weight/description/
class Solution:
  def __init__(self, w: List[int]):
    self.prefix = list(itertools.accumulate(w))

  def pickIndex(self) -> int:
    return bisect_left(self.prefix, random.random() * self.prefix[-1])


# Link: https://leetcode.com/problems/permutation-in-string/description/
class Solution:
  def checkInclusion(self, s1: str, s2: str) -> bool:
    count = collections.Counter(s1)
    required = len(s1)

    for r, c in enumerate(s2):
      count[c] -= 1
      if count[c] >= 0:
        required -= 1
      if r >= len(s1):  # The window is oversized.
        count[s2[r - len(s1)]] += 1
        if count[s2[r - len(s1)]] > 0:
          required += 1
      if required == 0:
        return True

    return False


# Link: https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/description/
class Solution:
  def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
    ans = [0] * len(nums)
    numAndIndexes = sorted([(num, i) for i, num in enumerate(nums)])
    # [[(num, index)]], where the difference between in each pair in each
    # `[(num, index)]` group <= `limit`
    numAndIndexesGroups: List[List[Tuple[int, int]]] = []

    for numAndIndex in numAndIndexes:
      if not numAndIndexesGroups or numAndIndex[0] - numAndIndexesGroups[-1][-1][0] > limit:
        # Start a new group.
        numAndIndexesGroups.append([numAndIndex])
      else:
        # Append to the existing group.
        numAndIndexesGroups[-1].append(numAndIndex)

    for numAndIndexesGroup in numAndIndexesGroups:
      sortedNums = [num for num, _ in numAndIndexesGroup]
      sortedIndices = sorted([index for _, index in numAndIndexesGroup])
      for num, index in zip(sortedNums, sortedIndices):
        ans[index] = num

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram-ii/description/
class Solution:
  def minSteps(self, s: str, t: str) -> int:
    count = collections.Counter(s)
    count.subtract(collections.Counter(t))
    return sum([abs(c) for c in count.values()])


# Link: https://leetcode.com/problems/wiggle-sort-ii/description/
class Solution:
  def wiggleSort(self, nums: List[int]) -> None:
    n = len(nums)
    median = self._findKthLargest(nums, (n + 1) // 2)

    def A(i: int):
      return (1 + 2 * i) % (n | 1)

    i = 0
    j = 0
    k = n - 1

    while i <= k:
      if nums[A(i)] > median:
        nums[A(i)], nums[A(j)] = nums[A(j)], nums[A(i)]
        i, j = i + 1, j + 1
      elif nums[A(i)] < median:
        nums[A(i)], nums[A(k)] = nums[A(k)], nums[A(i)]
        k -= 1
      else:
        i += 1

  # Same as 215. Kth Largest Element in an Array
  def _findKthLargest(self, nums: List[int], k: int) -> int:
    def quickSelect(l: int, r: int, k: int) -> int:
      randIndex = random.randint(0, r - l) + l
      nums[randIndex], nums[r] = nums[r], nums[randIndex]
      pivot = nums[r]

      nextSwapped = l
      for i in range(l, r):
        if nums[i] >= pivot:
          nums[nextSwapped], nums[i] = nums[i], nums[nextSwapped]
          nextSwapped += 1
      nums[nextSwapped], nums[r] = nums[r], nums[nextSwapped]

      count = nextSwapped - l + 1  # Number of nums >= pivot
      if count == k:
        return nums[nextSwapped]
      if count > k:
        return quickSelect(l, nextSwapped - 1, k)
      return quickSelect(nextSwapped + 1, r, k - count)

    return quickSelect(0, len(nums) - 1, k)


# Link: https://leetcode.com/problems/maximum-number-of-points-with-cost/description/
class Solution:
  def maxPoints(self, points: List[List[int]]) -> int:
    n = len(points[0])
    # dp[j] := the maximum number of points you can have if points[i][j] is the
    # most recent cell you picked
    dp = [0] * n

    for row in points:
      leftToRight = [0] * n
      runningMax = 0
      for j in range(n):
        runningMax = max(runningMax - 1, dp[j])
        leftToRight[j] = runningMax

      rightToLeft = [0] * n
      runningMax = 0
      for j in range(n - 1, - 1, -1):
        runningMax = max(runningMax - 1, dp[j])
        rightToLeft[j] = runningMax

      for j in range(n):
        dp[j] = max(leftToRight[j], rightToLeft[j]) + row[j]

    return max(dp)


# Link: https://leetcode.com/problems/cyclically-rotating-a-grid/description/
class Solution:
  def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
    m = len(grid)
    n = len(grid[0])
    t = 0  # the top
    l = 0  # the left
    b = m - 1  # the bottom
    r = n - 1  # the right

    while t < b and l < r:
      elementInThisLayer = 2 * (b - t + 1) + 2 * (r - l + 1) - 4
      netRotations = k % elementInThisLayer
      for _ in range(netRotations):
        topLeft = grid[t][l]
        for j in range(l, r):
          grid[t][j] = grid[t][j + 1]
        for i in range(t, b):
          grid[i][r] = grid[i + 1][r]
        for j in range(r, l, - 1):
          grid[b][j] = grid[b][j - 1]
        for i in range(b, t, -1):
          grid[i][l] = grid[i - 1][l]
        grid[t + 1][l] = topLeft
      t += 1
      l += 1
      b -= 1
      r -= 1

    return grid


# Link: https://leetcode.com/problems/minimum-increment-operations-to-make-array-beautiful/description/
class Solution:
  def minIncrementOperations(self, nums: List[int], k: int) -> int:
    # the minimum operations to increase nums[i - 3] and nums[0..i - 3)
    prev3 = 0
    # the minimum operations to increase nums[i - 2] and nums[0..i - 2)
    prev2 = 0
    # the minimum operations to increase nums[i - 1] and nums[0..i - 1)
    prev1 = 0

    for num in nums:
      dp = min(prev1, prev2, prev3) + max(0, k - num)
      prev3 = prev2
      prev2 = prev1
      prev1 = dp

    return min(prev1, prev2, prev3)


# Link: https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/description/
class Solution:
  def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
    def helper(l: int, r: int) -> int:
      n = len(nums)
      left = [0] * n
      summ = 0

      for i in range(n):
        summ += nums[i]
        if i >= l:
          summ -= nums[i - l]
        if i >= l - 1:
          left[i] = max(left[i - 1], summ) if i > 0 else summ

      right = [0] * n
      summ = 0

      for i in reversed(range(n)):
        summ += nums[i]
        if i <= n - r - 1:
          summ -= nums[i + r]
        if i <= n - r:
          right[i] = max(right[i + 1], summ) if i < n - 1 else summ

      return max(left[i] + right[i + 1] for i in range(n - 1))

    return max(helper(firstLen, secondLen), helper(secondLen, firstLen))


# Link: https://leetcode.com/problems/rearrange-words-in-a-sentence/description/
class Solution:
  def arrangeWords(self, text: str) -> str:
    words = text.split()
    count = collections.defaultdict(list)

    for word in words:
      count[len(word)].append(word.lower())

    c2 = OrderedDict(sorted(count.items()))

    ans = []

    for l in c2:
      for word in c2[l]:
        ans.append(word)

    ans[0] = ans[0].capitalize()

    return ' '.join(ans)


# Link: https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/
class Solution:
  def maxSubarrayLength(self, nums: List[int], k: int) -> int:
    ans = 0
    count = collections.Counter()

    l = 0
    for r, num in enumerate(nums):
      count[num] += 1
      while count[num] == k + 1:
        count[nums[l]] -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/minimum-impossible-or/description/
class Solution:
  def minImpossibleOR(self, nums: List[int]) -> int:
    ans = 1
    numsSet = set(nums)

    while ans in numsSet:
      ans <<= 1

    return ans


# Link: https://leetcode.com/problems/maximize-distance-to-closest-person/description/
class Solution:
  def maxDistToClosest(self, seats: List[int]) -> int:
    n = len(seats)
    ans = 0
    j = -1

    for i in range(n):
      if seats[i] == 1:
        ans = i if j == -1 else max(ans, (i - j) // 2)
        j = i

    return max(ans, n - j - 1)


# Link: https://leetcode.com/problems/minimum-cost-homecoming-of-a-robot-in-a-grid/description/
class Solution:
  def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
    ans = 0
    i, j = startPos
    x, y = homePos

    while i != x:
      i += 1 if i < x else -1
      ans += rowCosts[i]

    while j != y:
      j += 1 if j < y else -1
      ans += colCosts[j]

    return ans


# Link: https://leetcode.com/problems/corporate-flight-bookings/description/
class Solution:
  def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
    ans = [0] * n

    for booking in bookings:
      ans[booking[0] - 1] += booking[2]
      if booking[1] < n:
        ans[booking[1]] -= booking[2]

    for i in range(1, n):
      ans[i] += ans[i - 1]

    return ans


# Link: https://leetcode.com/problems/rearrange-array-to-maximize-prefix-score/description/
class Solution:
  def maxScore(self, nums: List[int]) -> int:
    return sum(num > 0
               for num in itertools.accumulate(sorted(nums, reverse=True)))


# Link: https://leetcode.com/problems/find-the-student-that-will-replace-the-chalk/description/
class Solution:
  def chalkReplacer(self, chalk: List[int], k: int) -> int:
    k %= sum(chalk)
    if k == 0:
      return 0

    for i, c in enumerate(chalk):
      k -= c
      if k < 0:
        return i


# Link: https://leetcode.com/problems/minimum-array-length-after-pair-removals/description/
class Solution:
  def minLengthAfterRemovals(self, nums: List[int]) -> int:
    n = len(nums)
    count = collections.Counter(nums)
    maxFreq = max(count.values())

    # The number with the maximum frequency cancel all the other numbers.
    if maxFreq <= n / 2:
      return n % 2
    # The number with the maximum frequency cancel all the remaining numbers.
    return maxFreq - (n - maxFreq)


# Link: https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/
class Solution:
  def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    summ = sum(nums)
    if summ % k != 0:
      return False

    target = summ // k  # the target sum of each subset
    if any(num > target for num in nums):
      return False

    def dfs(s: int, remainingGroups: int, currSum: int, used: int) -> bool:
      if remainingGroups == 0:
        return True
      if currSum > target:
        return False
      if currSum == target:  # Find a valid group, so fresh start.
        return dfs(0, remainingGroups - 1, 0, used)

      for i in range(s, len(nums)):
        if used >> i & 1:
          continue
        if dfs(i + 1, remainingGroups, currSum + nums[i], used | 1 << i):
          return True

      return False

    nums.sort(reverse=True)
    return dfs(0, k, 0, 0)


# Link: https://leetcode.com/problems/minimum-seconds-to-equalize-a-circular-array/description/
class Solution:
  def minimumSeconds(self, nums: List[int]) -> int:
    n = len(nums)
    ans = n
    numToIndices = collections.defaultdict(list)

    for i, num in enumerate(nums):
      numToIndices[num].append(i)

    def getSeconds(i: int, j: int) -> int:
      """Returns the number of seconds required to make nums[i..j] the same."""
      return (i - j) // 2

    for indices in numToIndices.values():
      seconds = getSeconds(indices[0] + n, indices[-1])
      for i in range(1, len(indices)):
        seconds = max(seconds, getSeconds(indices[i], indices[i - 1]))
      ans = min(ans, seconds)

    return ans


# Link: https://leetcode.com/problems/powx-n/description/
class Solution:
  def myPow(self, x: float, n: int) -> float:
    if n == 0:
      return 1
    if n < 0:
      return 1 / self.myPow(x, -n)
    if n & 1:
      return x * self.myPow(x, n - 1)
    return self.myPow(x * x, n // 2)


# Link: https://leetcode.com/problems/split-concatenated-strings/description/
class Solution:
  def splitLoopedString(self, strs: List[str]) -> str:
    ans = ''
    sortedStrs = [max(s, s[::-1]) for s in strs]

    for i, sortedStr in enumerate(sortedStrs):
      for s in (sortedStr, sortedStr[::-1]):
        for j in range(len(s) + 1):
          ans = max(
              ans, s[j:] + ''.join(sortedStrs[i + 1:] + sortedStrs[:i]) + s[:j])

    return ans


# Link: https://leetcode.com/problems/swap-for-longest-repeated-character-substring/description/
class Solution:
  def maxRepOpt1(self, text: str) -> int:
    count = collections.Counter(text)
    groups = [[c, len(list(group))]
              for c, group in itertools.groupby(text)]
    ans = max(min(length + 1, count[c]) for c, length in groups)

    for i in range(1, len(groups) - 1):
      if groups[i - 1][0] == groups[i + 1][0] and groups[i][1] == 1:
        ans = max(
            ans, min(groups[i - 1][1] + groups[i + 1][1] + 1, count[groups[i - 1][0]]))

    return ans


# Link: https://leetcode.com/problems/minimize-product-sum-of-two-arrays/description/
class Solution:
  def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
    return sum([a * b for a, b in zip(sorted(nums1), sorted(nums2, reverse=True))])


# Link: https://leetcode.com/problems/two-sum-bsts/description/
class BSTIterator:
  def __init__(self, root: Optional[TreeNode], leftToRight: bool):
    self.stack = []
    self.leftToRight = leftToRight
    self._pushUntilNone(root)

  def hasNext(self) -> bool:
    return len(self.stack) > 0

  def next(self) -> int:
    node = self.stack.pop()
    if self.leftToRight:
      self._pushUntilNone(node.right)
    else:
      self._pushUntilNone(node.left)
    return node.val

  def _pushUntilNone(self, root: Optional[TreeNode]):
    while root:
      self.stack.append(root)
      root = root.left if self.leftToRight else root.right


class Solution:
  def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
    bst1 = BSTIterator(root1, True)
    bst2 = BSTIterator(root2, False)

    l = bst1.next()
    r = bst2.next()
    while True:
      summ = l + r
      if summ == target:
        return True
      if summ < target:
        if not bst1.hasNext():
          return False
        l = bst1.next()
      else:
        if not bst2.hasNext():
          return False
        r = bst2.next()


# Link: https://leetcode.com/problems/encode-and-decode-tinyurl/description/
class Codec:
  alphabets = string.ascii_letters + '0123456789'
  urlToCode = {}
  codeToUrl = {}

  def encode(self, longUrl: str) -> str:
    while longUrl not in self.urlToCode:
      code = ''.join(random.choice(self.alphabets) for _ in range(6))
      if code not in self.codeToUrl:
        self.codeToUrl[code] = longUrl
        self.urlToCode[longUrl] = code
    return 'http://tinyurl.com/' + self.urlToCode[longUrl]

  def decode(self, shortUrl: str) -> str:
    return self.codeToUrl[shortUrl[-6:]]


# Link: https://leetcode.com/problems/minimum-height-trees/description/
class Solution:
  def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
    if n == 1 or not edges:
      return [0]

    ans = []
    graph = collections.defaultdict(set)

    for u, v in edges:
      graph[u].add(v)
      graph[v].add(u)

    for label, children in graph.items():
      if len(children) == 1:
        ans.append(label)

    while n > 2:
      n -= len(ans)
      nextLeaves = []
      for leaf in ans:
        u = next(iter(graph[leaf]))
        graph[u].remove(leaf)
        if len(graph[u]) == 1:
          nextLeaves.append(u)
      ans = nextLeaves

    return ans


# Link: https://leetcode.com/problems/hand-of-straights/description/
class Solution:
  def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
    count = collections.Counter(hand)

    for start in sorted(count):
      value = count[start]
      if value > 0:
        for i in range(start, start + groupSize):
          count[i] -= value
          if count[i] < 0:
            return False

    return True


# Link: https://leetcode.com/problems/minimum-absolute-sum-difference/description/
class Solution:
  def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
    ans = math.inf
    diffs = [abs(a - b) for a, b in zip(nums1, nums2)]
    sumDiff = sum(diffs)

    nums1.sort()

    for num, diff in zip(nums2, diffs):
      i = bisect.bisect_left(nums1, num)
      if i > 0:
        ans = min(ans, sumDiff - diff + abs(num - nums1[i - 1]))
      if i < len(nums1):
        ans = min(ans, sumDiff - diff + abs(num - nums1[i]))

    return ans % int(1e9 + 7)


# Link: https://leetcode.com/problems/number-of-good-binary-strings/description/
class Solution:
  def goodBinaryStrings(self, minLength: int, maxLength: int, oneGroup: int, zeroGroup: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of good binary strings with length i
    dp = [1] + [0] * maxLength

    for i in range(maxLength + 1):
      # There are good binary strings with length i, so we can append
      # consecutive 0s or 1s after it.
      if dp[i] > 0:
        appendZeros = i + zeroGroup
        if appendZeros <= maxLength:
          dp[appendZeros] += dp[i]
          dp[appendZeros] %= kMod
        appendOnes = i + oneGroup
        if appendOnes <= maxLength:
          dp[appendOnes] += dp[i]
          dp[appendOnes] %= kMod

    return sum(dp[minLength:]) % kMod


# Link: https://leetcode.com/problems/count-and-say/description/
class Solution:
  def countAndSay(self, n: int) -> str:
    ans = '1'

    for _ in range(n - 1):
      nxt = ''
      i = 0
      while i < len(ans):
        count = 1
        while i + 1 < len(ans) and ans[i] == ans[i + 1]:
          count += 1
          i += 1
        nxt += str(count) + ans[i]
        i += 1
      ans = nxt

    return ans


# Link: https://leetcode.com/problems/as-far-from-land-as-possible/description/
class Solution:
  def maxDistance(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    q = collections.deque()
    water = 0

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 0:
          water += 1
        else:
          q.append((i, j))

    if water == 0 or water == m * n:
      return -1

    ans = 0
    d = 0

    while q:
      for _ in range(len(q)):
        i, j = q.popleft()
        ans = d
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if grid[x][y] > 0:
            continue
          q.append((x, y))
          grid[x][y] = 2  # Mark as visited.
      d += 1

    return ans


# Link: https://leetcode.com/problems/domino-and-tromino-tiling/description/
class Solution:
  def numTilings(self, n: int) -> int:
    kMod = 1_000_000_007
    dp = [0, 1, 2, 5] + [0] * 997

    for i in range(4, n + 1):
      dp[i] = 2 * dp[i - 1] + dp[i - 3]

    return dp[n] % kMod


# Link: https://leetcode.com/problems/find-the-score-of-all-prefixes-of-an-array/description/
class Solution:
  def findPrefixScore(self, nums: List[int]) -> List[int]:
    conver = []
    maxi = 0

    for num in nums:
      maxi = max(maxi, num)
      conver.append(num + maxi)

    return itertools.accumulate(conver)


# Link: https://leetcode.com/problems/longest-univalue-path/description/
class Solution:
  def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def longestUnivaluePathDownFrom(root: Optional[TreeNode]) -> int:
      nonlocal ans
      if not root:
        return 0

      l = longestUnivaluePathDownFrom(root.left)
      r = longestUnivaluePathDownFrom(root.right)
      arrowLeft = l + 1 if root.left and root.left.val == root.val else 0
      arrowRight = r + 1 if root.right and root.right.val == root.val else 0
      ans = max(ans, arrowLeft + arrowRight)
      return max(arrowLeft, arrowRight)

    longestUnivaluePathDownFrom(root)
    return ans


# Link: https://leetcode.com/problems/unique-word-abbreviation/description/
class ValidWordAbbr:
  def __init__(self, dictionary: List[str]):
    self.dict = set(dictionary)
    # T := unique, F := not unique
    self.abbrUnique = {}

    for word in self.dict:
      abbr = self._getAbbr(word)
      self.abbrUnique[abbr] = abbr not in self.abbrUnique

  def isUnique(self, word: str) -> bool:
    abbr = self._getAbbr(word)
    return abbr not in self.abbrUnique or self.abbrUnique[abbr] and word in self.dict

  def _getAbbr(self, s: str) -> str:
    n = len(s)
    if n <= 2:
      return s
    return s[0] + str(n - 2) + s[-1]


# Link: https://leetcode.com/problems/strictly-palindromic-number/description/
class Solution:
  def isStrictlyPalindromic(self, n: int) -> bool:
    return False


# Link: https://leetcode.com/problems/construct-the-longest-new-string/description/
class Solution:
  def longestString(self, x: int, y: int, z: int) -> int:
    # 'AB' can always be easily appended within the string.
    # Alternating 'AA' and 'BB' can be appended, creating a pattern like 'AABB'
    # If x == y, we repeat the pattern 'AABBAABB...AABB'.
    # If x != y, the pattern becomes 'AABBAABB...AABBAA' or 'BBAABBAABB...AABB'
    mini = min(x, y)
    if x == y:
      return (mini * 2 + z) * 2
    return (mini * 2 + 1 + z) * 2


# Link: https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make/description/
class Solution:
  def getMaximumConsecutive(self, coins: List[int]) -> int:
    ans = 1  # the next value we want to make

    for coin in sorted(coins):
      if coin > ans:
        return ans
      ans += coin

    return ans


# Link: https://leetcode.com/problems/find-latest-group-of-size-m/description/
class Solution:
  def findLatestStep(self, arr: List[int], m: int) -> int:
    if len(arr) == m:
      return len(arr)

    ans = -1
    step = 0
    # sizes[i] := the size of the group starting from i or ending in i
    # (1-indexed)
    sizes = [0] * (len(arr) + 2)

    for i in arr:
      step += 1
      # In the previous step, there exists a group with a size of m.
      if sizes[i - 1] == m or sizes[i + 1] == m:
        ans = step - 1
      head = i - sizes[i - 1]
      tail = i + sizes[i + 1]
      sizes[head] = tail - head + 1
      sizes[tail] = tail - head + 1

    return ans


# Link: https://leetcode.com/problems/two-city-scheduling/description/
class Solution:
  def twoCitySchedCost(self, costs: List[List[int]]) -> int:
    n = len(costs) // 2

    # How much money can we save if we fly a person to A instead of B?
    # To save money, we should
    #   1. Fly the person with the maximum saving to A.
    #   2. Fly the person with the minimum saving to B.

    # Sort `costs` in descending order by the money saved if we fly a person
    # to A instead of B.
    costs.sort(key=lambda x: x[0] - x[1])
    return sum(costs[i][0] + costs[i + n][1] for i in range(n))


# Link: https://leetcode.com/problems/add-two-numbers-ii/description/
class Solution:
  def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    stack1 = []
    stack2 = []

    while l1:
      stack1.append(l1)
      l1 = l1.next

    while l2:
      stack2.append(l2)
      l2 = l2.next

    head = None
    carry = 0

    while carry or stack1 or stack2:
      if stack1:
        carry += stack1.pop().val
      if stack2:
        carry += stack2.pop().val
      node = ListNode(carry % 10)
      node.next = head
      head = node
      carry //= 10

    return head


# Link: https://leetcode.com/problems/rotate-array/description/
class Solution:
  def rotate(self, nums: List[int], k: int) -> None:
    k %= len(nums)
    self.reverse(nums, 0, len(nums) - 1)
    self.reverse(nums, 0, k - 1)
    self.reverse(nums, k, len(nums) - 1)

  def reverse(self, nums: List[int], l: int, r: int) -> None:
    while l < r:
      nums[l], nums[r] = nums[r], nums[l]
      l += 1
      r -= 1


# Link: https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/
class Solution:
  def maximumSum(self, nums: List[int]) -> int:
    kMax = 9 * 9  # 999,999,999
    ans = -1
    count = [[] for _ in range(kMax + 1)]

    for num in nums:
      count[self._getDigitSum(num)].append(num)

    for groupNums in count:
      if len(groupNums) < 2:
        continue
      groupNums.sort(reverse=True)
      ans = max(ans, groupNums[0] + groupNums[1])

    return ans

  def _getDigitSum(self, num: int) -> int:
    return sum(int(digit) for digit in str(num))


# Link: https://leetcode.com/problems/most-frequent-subtree-sum/description/
class Solution:
  def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
      return []

    count = collections.Counter()

    def dfs(root: Optional[TreeNode]) -> int:
      if not root:
        return 0

      summ = root.val + dfs(root.left) + dfs(root.right)
      count[summ] += 1
      return summ

    dfs(root)
    maxFreq = max(count.values())
    return [summ for summ in count if count[summ] == maxFreq]


# Link: https://leetcode.com/problems/count-number-of-texts/description/
class Solution:
  def countTexts(self, pressedKeys: str) -> int:
    kMod = 1_000_000_007
    n = len(pressedKeys)
    # dp[i] := the number of possible text messages of pressedKeys[i..n)
    dp = [0] * n + [1]

    def isSame(s: str, i: int, k: int) -> bool:
      """Returns true if s[i..i + k) are the same digits."""
      if i + k > len(s):
        return False
      for j in range(i + 1, i + k):
        if s[j] != s[i]:
          return False
      return True

    for i in reversed(range(n)):
      dp[i] = dp[i + 1]
      if isSame(pressedKeys, i, 2):
        dp[i] += dp[i + 2]
      if isSame(pressedKeys, i, 3):
        dp[i] += dp[i + 3]
      if (pressedKeys[i] == '7' or pressedKeys[i] == '9') and \
              isSame(pressedKeys, i, 4):
        dp[i] += dp[i + 4]
      dp[i] %= kMod

    return dp[0]


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-sort-a-binary-tree-by-level/description/
class Solution:
  def minimumOperations(self, root: Optional[TreeNode]) -> int:
    ans = 0
    q = collections.deque([root])

    # e.g. vals = [7, 6, 8, 5]
    # [2, 1, 3, 0]: Initialize the ids based on the order of vals.
    # [3, 1, 2, 0]: Swap 2 with 3, so 2 is in the right place (i == ids[i]).
    # [0, 1, 2, 3]: Swap 3 with 0, so 3 is in the right place.
    while q:
      vals = []
      for _ in range(len(q)):
        node = q.popleft()
        vals.append(node.val)
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      # O(n^2logn), which is not great and leads to TLE.
      ids = [sorted(vals).index(val) for val in vals]
      for i in range(len(ids)):
        while ids[i] != i:
          j = ids[i]
          ids[i] = ids[j]
          ids[j] = j
          ans += 1

    return ans


# Link: https://leetcode.com/problems/design-a-leaderboard/description/
class Leaderboard:
  def __init__(self):
    self.idToScore = collections.Counter()

  def addScore(self, playerId: int, score: int) -> None:
    self.idToScore[playerId] += score

  def top(self, K: int) -> int:
    return sum(score for _, score in self.idToScore.most_common(K))

  def reset(self, playerId: int) -> None:
    del self.idToScore[playerId]


# Link: https://leetcode.com/problems/count-number-of-bad-pairs/description/
class Solution:
  def countBadPairs(self, nums: List[int]) -> int:
    ans = 0
    count = collections.Counter()  # (nums[i] - i)

    for i, num in enumerate(nums):
      #     count[nums[i] - i] := the number of good pairs
      # i - count[nums[i] - i] := the number of bad pairs
      ans += i - count[num - i]
      count[num - i] += 1

    return ans


# Link: https://leetcode.com/problems/all-divisions-with-the-highest-score-of-a-binary-array/description/
class Solution:
  def maxScoreIndices(self, nums: List[int]) -> List[int]:
    zeros = nums.count(0)
    ones = len(nums) - zeros
    ans = [0]  # the division at index 0
    leftZeros = 0
    leftOnes = 0
    maxScore = ones  # `leftZeros` + `rightOnes`

    for i, num in enumerate(nums):
      leftZeros += num == 0
      leftOnes += num == 1
      rightOnes = ones - leftOnes
      score = leftZeros + rightOnes
      if maxScore == score:
        ans.append(i + 1)
      elif maxScore < score:
        maxScore = score
        ans = [i + 1]

    return ans


# Link: https://leetcode.com/problems/prime-pairs-with-target-sum/description/
class Solution:
  def findPrimePairs(self, n: int) -> List[List[int]]:
    isPrime = self._sieveEratosthenes(n + 1)
    return [[i, n - i] for i in range(2, n // 2 + 1)
            if isPrime[i] and isPrime[n - i]]

  def _sieveEratosthenes(self, n: int) -> List[bool]:
    isPrime = [True] * n
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n**0.5) + 1):
      if isPrime[i]:
        for j in range(i * i, n, i):
          isPrime[j] = False
    return isPrime
j


# Link: https://leetcode.com/problems/connecting-cities-with-minimum-cost/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self.find(u)
    j = self.find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def minimumCost(self, n: int, connections: List[List[int]]) -> int:
    ans = 0
    uf = UnionFind(n + 1)

    # Sort by cost.
    connections.sort(key=lambda x: x[2])

    for u, v, cost in connections:
      if uf.find(u) == uf.find(v):
        continue
      uf.unionByRank(u, v)
      ans += cost

    root = uf.find(1)
    if any(uf.find(i) != root for i in range(1, n + 1)):
      return -1

    return ans


# Link: https://leetcode.com/problems/extra-characters-in-a-string/description/
class Solution:
  # Similar to 139. Word Break
  def minExtraChar(self, s: str, dictionary: List[str]) -> int:
    n = len(s)
    dictionarySet = set(dictionary)
    # dp[i] := the minimum extra letters if breaking up s[0..i) optimally
    dp = [0] + [n] * n

    for i in range(1, n + 1):
      for j in range(i):
        if s[j:i] in dictionarySet:
          dp[i] = min(dp[i], dp[j])
        else:
          dp[i] = min(dp[i], dp[j] + i - j)

    return dp[n]


# Link: https://leetcode.com/problems/eliminate-maximum-number-of-monsters/description/
class Solution:
  def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
    for i, arrivalTime in enumerate(sorted([(d - 1) // s for d, s in zip(dist, speed)])):
      if i > arrivalTime:
        return i
    return len(dist)


# Link: https://leetcode.com/problems/remove-interval/description/
class Solution:
  def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
    ans = []

    for a, b in intervals:
      if a >= toBeRemoved[1] or b <= toBeRemoved[0]:
        ans.append([a, b])
      else:  # a < toBeRemoved[1] and b > toBeRemoved[0]
        if a < toBeRemoved[0]:
          ans.append([a, toBeRemoved[0]])
        if b > toBeRemoved[1]:
          ans.append([toBeRemoved[1], b])

    return ans


# Link: https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/description/
class Solution:
  def minimumCardPickup(self, cards: List[int]) -> int:
    ans = math.inf
    lastSeen = {}

    for i, card in enumerate(cards):
      if card in lastSeen:
        ans = min(ans, i - lastSeen[card] + 1)
      lastSeen[card] = i

    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/get-biggest-three-rhombus-sums-in-a-grid/description/
from sortedcontainers import SortedSet


class Solution:
  def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
    m = len(grid)
    n = len(grid[0])
    sums = SortedSet()

    for i in range(m):
      for j in range(n):
        sz = 0
        while i + sz < m and i - sz >= 0 and j + 2 * sz < n:
          summ = grid[i][j] if sz == 0 else self._getSum(grid, i, j, sz)
          sums.add(summ)
          if len(sums) > 3:
            sums.pop(0)
          sz += 1

    return sums

  def _getSum(self, grid: List[List[int]], i: int, j: int, sz: int) -> int:
    """
    Returns the sum of the rhombus, where the top grid is (i, j) and the edge
    size is `sz`.
    """
    x = i
    y = j
    summ = 0

    # Go left down.
    for _ in range(sz):
      x -= 1
      y += 1
      summ += grid[x][y]

    # Go right down.
    for _ in range(sz):
      x += 1
      y += 1
      summ += grid[x][y]

    # Go right up.
    for _ in range(sz):
      x += 1
      y -= 1
      summ += grid[x][y]

    # Go left up.
    for _ in range(sz):
      x -= 1
      y -= 1
      summ += grid[x][y]

    return summ


# Link: https://leetcode.com/problems/split-array-into-maximum-number-of-subarrays/description/
class Solution:
  def maxSubarrays(self, nums: List[int]) -> int:
    ans = 0
    score = 0

    for num in nums:
      score = num if score == 0 else score & num
      if score == 0:
        ans += 1

    return max(1, ans)


# Link: https://leetcode.com/problems/find-a-peak-element-ii/description/
class Solution:
  def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
    l = 0
    r = len(mat) - 1

    while l < r:
      m = (l + r) // 2
      if max(mat[m]) >= max(mat[m + 1]):
        r = m
      else:
        l = m + 1

    return [l, mat[l].index(max(mat[l]))]


# Link: https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/description/
class Solution:
  def minDifference(self, nums: List[int]) -> int:
    n = len(nums)
    if n < 5:
      return 0

    ans = math.inf

    nums.sort()

    # 1. Change nums[0..i) to nums[i].
    # 2. Change nums[n - 3 + i..n) to nums[n - 4 + i].
    for i in range(4):
      ans = min(ans, nums[n - 4 + i] - nums[i])

    return ans


# Link: https://leetcode.com/problems/maximum-alternating-subsequence-sum/description/
class Solution:
  def maxAlternatingSum(self, nums: List[int]) -> int:
    even = 0  # the maximum alternating sum ending in an even index
    odd = 0  # the maximum alternating sum ending in an odd index

    for num in nums:
      even = max(even, odd + num)
      odd = even - num

    return even


# Link: https://leetcode.com/problems/minimum-health-to-beat-game/description/
class Solution:
  def minimumHealth(self, damage: List[int], armor: int) -> int:
    return 1 + sum(damage) - min(max(damage), armor)


# Link: https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/description/
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class ArrayReader:
#   def get(self, index: int) -> int:

class Solution:
  def search(self, reader: 'ArrayReader', target: int) -> int:
    l = bisect.bisect_left(range(10**4), target,
                           key=lambda m: reader.get(m))
    return l if reader.get(l) == target else -1


# Link: https://leetcode.com/problems/jump-game-ii/description/
class Solution:
  def jump(self, nums: List[int]) -> int:
    ans = 0
    end = 0
    farthest = 0

    # Start an implicit BFS.
    for i in range(len(nums) - 1):
      farthest = max(farthest, i + nums[i])
      if farthest >= len(nums) - 1:
        ans += 1
        break
      if i == end:      # Visited all the items on the current level.
        ans += 1        # Increment the level.
        end = farthest  # Make the queue size for the next level.

    return ans


# Link: https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/
class Solution:
  def minCost(self, colors: str, neededTime: List[int]) -> int:
    ans = 0
    maxNeededTime = neededTime[0]

    for i in range(1, len(colors)):
      if colors[i] == colors[i - 1]:
        ans += min(maxNeededTime, neededTime[i])
        # For each continuous group, Bob needs to remove every balloon except
        # the one with the maximum `neededTime`. So, he should hold the balloon
        # with the highest `neededTime` in his hand.
        maxNeededTime = max(maxNeededTime, neededTime[i])
      else:
        # If the current balloon is different from the previous one, discard
        # the balloon from the previous group and hold the new one in hand.
        maxNeededTime = neededTime[i]

    return ans


# Link: https://leetcode.com/problems/design-underground-system/description/
class UndergroundSystem:
  def __init__(self):
    # {id: (stationName, time)}
    self.checkIns = {}
    # {route: (numTrips, totalTime)}
    self.checkOuts = collections.defaultdict(lambda: [0, 0])

  def checkIn(self, id: int, stationName: str, t: int) -> None:
    self.checkIns[id] = (stationName, t)

  def checkOut(self, id: int, stationName: str, t: int) -> None:
    startStation, startTime = self.checkIns.pop(id)
    route = (startStation, stationName)
    self.checkOuts[route][0] += 1
    self.checkOuts[route][1] += t - startTime

  def getAverageTime(self, startStation: str, endStation: str) -> float:
    numTrips, totalTime = self.checkOuts[(startStation, endStation)]
    return totalTime / numTrips


# Link: https://leetcode.com/problems/custom-sort-string/description/
class Solution:
  def customSortString(self, order: str, s: str) -> str:
    ans = ""
    count = [0] * 26

    for c in s:
      count[ord(c) - ord('a')] += 1

    for c in order:
      while count[ord(c) - ord('a')] > 0:
        ans += c
        count[ord(c) - ord('a')] -= 1

    for c in string.ascii_lowercase:
      for _ in range(count[ord(c) - ord('a')]):
        ans += c

    return ans


# Link: https://leetcode.com/problems/check-if-word-can-be-placed-in-crossword/description/
class Solution:
  def placeWordInCrossword(self, board: List[List[str]], word: str) -> bool:
    for x in board, zip(*board):
      for row in x:
        for token in ''.join(row).split('#'):
          for letters in word, word[::-1]:
            if len(token) == len(letters):
              if all(c in (' ', letter) for c, letter in zip(token, letters)):
                return True
    return False


# Link: https://leetcode.com/problems/high-access-employees/description/
class Solution:
  def findHighAccessEmployees(self, access_times: List[List[str]]) -> List[str]:
    ans = set()

    access_times.sort()

    for i in range(len(access_times) - 2):
      name = access_times[i][0]
      if name in ans:
        continue
      if name != access_times[i + 2][0]:
        continue
      if int(access_times[i + 2][1]) - int(access_times[i][1]) < 100:
        ans.add(name)

    return list(ans)


# Link: https://leetcode.com/problems/maximize-total-tastiness-of-purchased-fruits/description/
class Solution:
  def maxTastiness(self, price: List[int], tastiness: List[int], maxAmount: int, maxCoupons: int) -> int:
    # dp[j][k] := the maximum tastiness of price so far with j amount of money and k coupons
    dp = [[0] * (maxCoupons + 1) for _ in range(maxAmount + 1)]

    for p, t in zip(price, tastiness):
      for j in range(maxAmount, p // 2 - 1, -1):
        for k in range(maxCoupons, -1, -1):
          buyWithCoupon = 0 if k == 0 else dp[j - p // 2][k - 1] + t
          buyWithoutCoupon = 0 if j < p else dp[j - p][k] + t
          dp[j][k] = max(dp[j][k], buyWithCoupon, buyWithoutCoupon)

    return dp[maxAmount][maxCoupons]


# Link: https://leetcode.com/problems/maximize-total-tastiness-of-purchased-fruits/description/
class Solution:
  def maxTastiness(self, price: List[int], tastiness: List[int], maxAmount: int, maxCoupons: int) -> int:
    n = len(price)
    # dp[i][j][k] := the maximum tastiness of first i price with j amount of money and k coupons
    dp = [[[0] * (maxCoupons + 1)
           for j in range(maxAmount + 1)]
          for i in range(n + 1)]

    for i in range(1, n + 1):
      # 1-indexed
      currPrice = price[i - 1]
      currTastiness = tastiness[i - 1]
      for amount in range(maxAmount + 1):
        for coupon in range(maxCoupons + 1):
          # Case 1: Don't buy, the tastiness will be the same as the first i - 1 price.
          dp[i][amount][coupon] = dp[i - 1][amount][coupon]

          # Case 2: Buy without coupon if have enough money.
          if amount >= currPrice:
            dp[i][amount][coupon] = max(
                dp[i][amount][coupon],
                dp[i - 1][amount - currPrice][coupon] + currTastiness)

          # Case 3: Buy with coupon if have coupon and enough money.
          if coupon > 0 and amount >= currPrice // 2:
            dp[i][amount][coupon] = max(
                dp[i][amount][coupon],
                dp[i - 1][amount - currPrice // 2][coupon - 1] + currTastiness)

    return dp[n][maxAmount][maxCoupons]


# Link: https://leetcode.com/problems/minimum-moves-to-capture-the-queen/description/
class Solution:
  def minMovesToCaptureTheQueen(self, a: int, b: int, c: int, d: int, e: int, f: int) -> int:
    # The rook is in the same row as the queen.
    if a == e:
      # The bishop blocks the rook or not.
      return 2 if c == a and (b < d < f or b > d > f) else 1
    # The rook is in the same column as the queen.
    if b == f:
      # The bishop blocks the rook or not.
      return 2 if d == f and (a < c < e or a > c > e) else 1
    # The bishop is in the same up-diagonal as the queen.
    if c + d == e + f:
      # The rook blocks the bishop or not.
      return 2 if a + b == c + d and (c < a < e or c > a > e) else 1
    # The bishop is in the same down-diagonal as the queen.
    if c - d == e - f:
      # The rook blocks the bishop or not.
      return 2 if a - b == c - d and (c < a < e or c > a > e) else 1
    # The rook can always get the green in two steps.
    return 2


# Link: https://leetcode.com/problems/find-polygon-with-the-largest-perimeter/description/
class Solution:
  def largestPerimeter(self, nums: List[int]) -> int:
    prefix = sum(nums)

    for num in sorted(nums, reverse=True):
      prefix -= num
      # Let `num` be the longest side. Check if the sum of all the edges with
      # length no longer than `num` > `num``.
      if prefix > num:
        return prefix + num

    return -1


# Link: https://leetcode.com/problems/plus-one-linked-list/description/
class Solution:
  def plusOne(self, head: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    dummy.next = head

    while head:
      if head.val != 9:
        curr = head
      head = head.next
    # `curr` now points to the rightmost non-9 node.

    curr.val += 1
    while curr.next:
      curr.next.val = 0
      curr = curr.next

    return dummy.next if dummy.val == 0 else dummy


# Link: https://leetcode.com/problems/plus-one-linked-list/description/
class Solution:
  def plusOne(self, head: ListNode) -> ListNode:
    if not head:
      return ListNode(1)
    if self._addOne(head) == 1:
      return ListNode(1, head)
    return head

  def _addOne(self, node: ListNode) -> int:
    carry = self._addOne(node.next) if node.next else 1
    summ = node.val + carry
    node.val = summ % 10
    return summ // 10


# Link: https://leetcode.com/problems/intervals-between-identical-elements/description/
class Solution:
  def getDistances(self, arr: List[int]) -> List[int]:
    prefix = [0] * len(arr)
    suffix = [0] * len(arr)
    numToIndices = collections.defaultdict(list)

    for i, a in enumerate(arr):
      numToIndices[a].append(i)

    for indices in numToIndices.values():
      for i in range(1, len(indices)):
        currIndex = indices[i]
        prevIndex = indices[i - 1]
        prefix[currIndex] += prefix[prevIndex] + i * (currIndex - prevIndex)
      for i in range(len(indices) - 2, -1, -1):
        currIndex = indices[i]
        prevIndex = indices[i + 1]
        suffix[currIndex] += suffix[prevIndex] + \
            (len(indices) - i - 1) * (prevIndex - currIndex)

    return [p + s for p, s in zip(prefix, suffix)]


# Link: https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/description/
class Solution:
  def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
    l = 1
    r = max(quantities)

    def numStores(m: int) -> int:
      return sum((q - 1) // m + 1 for q in quantities)

    while l < r:
      m = (l + r) // 2
      if numStores(m) <= n:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/min-cost-to-connect-all-points/description/
class Solution:
  def minCostConnectPoints(self, points: List[int]) -> int:
    # dist[i] := the minimum distance to connect the points[i]
    dist = [math.inf] * len(points)
    ans = 0

    for i in range(len(points) - 1):
      for j in range(i + 1, len(points)):
        # Try to connect the points[i] with the points[j].
        dist[j] = min(dist[j], abs(points[i][0] - points[j][0]) +
                      abs(points[i][1] - points[j][1]))
        # Swap the points[j] (the point with the mnimum distance) with the
        # points[i + 1].
        if dist[j] < dist[i + 1]:
          points[j], points[i + 1] = points[i + 1], points[j]
          dist[j], dist[i + 1] = dist[i + 1], dist[j]
      ans += dist[i + 1]

    return ans


# Link: https://leetcode.com/problems/insertion-sort-list/description/
class Solution:
  def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    prev = dummy  # the last and thus largest of the sorted list

    while head:  # the current inserting node
      next = head.next  # Cache the next inserting node.
      if prev.val >= head.val:
        prev = dummy  # Move `prev` to the front.
      while prev.next and prev.next.val < head.val:
        prev = prev.next
      head.next = prev.next
      prev.next = head
      head = next  # Update the current inserting node.

    return dummy.next


# Link: https://leetcode.com/problems/append-characters-to-string-to-make-subsequence/description/
class Solution:
  def appendCharacters(self, s: str, t: str) -> int:
    i = 0  # t's index

    for c in s:
      if c == t[i]:
        i += 1
        if i == len(t):
          return 0

    return len(t) - i


# Link: https://leetcode.com/problems/my-calendar-i/description/
class MyCalendar:
  def __init__(self):
    self.timeline = []

  def book(self, start: int, end: int) -> bool:
    for s, e in self.timeline:
      if max(start, s) < min(end, e):
        return False
    self.timeline.append((start, end))
    return True


# Link: https://leetcode.com/problems/my-calendar-i/description/
class Node:
  def __init__(self, start: int, end: int):
    self.start = start
    self.end = end
    self.left = None
    self.right = None


class Tree:
  def __init__(self):
    self.root = None

  def insert(self, node: Node, root: Node = None) -> bool:
    if not root:
      if not self.root:
        self.root = node
        return True
      else:
        root = self.root

    if node.start >= root.end:
      if not root.right:
        root.right = node
        return True
      return self.insert(node, root.right)
    elif node.end <= root.start:
      if not root.left:
        root.left = node
        return True
      return self.insert(node, root.left)
    else:
      return False


class MyCalendar:
  def __init__(self):
    self.tree = Tree()

  def book(self, start: int, end: int) -> bool:
    return self.tree.insert(Node(start, end))


# Link: https://leetcode.com/problems/maximum-earnings-from-taxi/description/
class Solution:
  def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
    startToEndAndEarns = [[] for _ in range(n)]
    # dp[i] := the maximum dollars you can earn starting at i
    dp = [0] * (n + 1)

    for start, end, tip in rides:
      earn = end - start + tip
      startToEndAndEarns[start].append((end, earn))

    for i in range(n - 1, 0, -1):
      dp[i] = dp[i + 1]
      for end, earn in startToEndAndEarns[i]:
        dp[i] = max(dp[i], dp[end] + earn)

    return dp[1]


# Link: https://leetcode.com/problems/maximum-earnings-from-taxi/description/
class Solution:
  def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
    endToStartAndEarns = [[] for _ in range(n + 1)]
    # dp[i] := the maximum dollars you can earn starting at i
    dp = [0] * (n + 1)

    for start, end, tip in rides:
      earn = end - start + tip
      endToStartAndEarns[end].append((start, earn))

    for i in range(1, n + 1):
      dp[i] = dp[i - 1]
      for start, earn in endToStartAndEarns[i]:
        dp[i] = max(dp[i], dp[start] + earn)

    return dp[n]


# Link: https://leetcode.com/problems/image-overlap/description/
class Solution:
  def largestOverlap(self, A: List[List[int]], B: List[List[int]]) -> int:
    n = len(A)
    magic = 100
    onesA = []
    onesB = []
    dict = collections.defaultdict(int)

    for i in range(n):
      for j in range(n):
        if A[i][j] == 1:
          onesA.append([i, j])
        if B[i][j] == 1:
          onesB.append([i, j])

    for a in onesA:
      for b in onesB:
        dict[(a[0] - b[0]) * magic + (a[1] - b[1])] += 1

    return max(dict.values()) if dict else 0


# Link: https://leetcode.com/problems/print-binary-tree/description/
class Solution:
  def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
    def maxHeight(root: Optional[TreeNode]) -> int:
      if not root:
        return 0
      return 1 + max(maxHeight(root.left), maxHeight(root.right))

    def dfs(root: Optional[TreeNode], row: int, left: int, right: int) -> None:
      if not root:
        return

      mid = (left + right) // 2
      ans[row][mid] = str(root.val)
      dfs(root.left, row + 1, left, mid - 1)
      dfs(root.right, row + 1, mid + 1, right)

    m = maxHeight(root)
    n = pow(2, m) - 1
    ans = [[''] * n for _ in range(m)]
    dfs(root, 0, 0, len(ans[0]) - 1)
    return ans


# Link: https://leetcode.com/problems/rotating-the-box/description/
class Solution:
  def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
    m = len(box)
    n = len(box[0])
    rotated = [['.'] * m for _ in range(n)]

    for i in range(m):
      k = n - 1
      for j in reversed(range(n)):
        if box[i][j] != '.':
          if box[i][j] == '*':
            k = j
          rotated[k][m - i - 1] = box[i][j]
          k -= 1

    return [''.join(row) for row in rotated]


# Link: https://leetcode.com/problems/tuple-with-same-product/description/
class Solution:
  def tupleSameProduct(self, nums: List[int]) -> int:
    # nums of ways to arrange (a, b) = 2
    # nums of ways to arrange (c, d) = 2
    # nums of ways to arrange (a, b), (c, d) = 2^3 = 8
    ans = 0
    count = collections.Counter()

    for i in range(len(nums)):
      for j in range(i):
        prod = nums[i] * nums[j]
        ans += count[prod] * 8
        count[prod] += 1

    return ans


# Link: https://leetcode.com/problems/reordered-power-of-2/description/
class Solution:
  def reorderedPowerOf2(self, n: int) -> bool:
    count = collections.Counter(str(n))
    return any(Counter(str(1 << i)) == count for i in range(30))


# Link: https://leetcode.com/problems/maximum-matching-of-players-with-trainers/description/
class Solution:
  def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
    ans = 0

    players.sort()
    trainers.sort()

    for i, trainer in enumerate(trainers):
      if players[ans] <= trainers[i]:
        ans += 1
        if ans == len(players):
          return ans

    return ans
  



# Link: https://leetcode.com/problems/check-if-there-is-a-path-with-equal-number-of-0s-and-1s/description/
class Solution:
  def isThereAPath(self, grid: List[List[int]]) -> bool:
    m = len(grid)
    n = len(grid[0])
    if m + n - 1 & 1:
      return False

    @functools.lru_cache(None)
    def dp(i: int, j: int, summ: int) -> bool:
      """
      Returns 1 if there's a path to grid[i][j] s.t.
      `summ` = (the number of 0s - the number of 1s).
      """
      if i == m or j == n:
        return False
      summ += 1 if grid[i][j] == 0 else -1
      if i == m - 1 and j == n - 1:
        return summ == 0
      return dp(i + 1, j, summ) or dp(i, j + 1, summ)

    return dp(0, 0, 0)


# Link: https://leetcode.com/problems/house-robber-ii/description/
class Solution:
  def rob(self, nums: List[int]) -> int:
    if not nums:
      return 0
    if len(nums) < 2:
      return nums[0]

    def rob(l: int, r: int) -> int:
      dp1 = 0
      dp2 = 0

      for i in range(l, r + 1):
        temp = dp1
        dp1 = max(dp1, dp2 + nums[i])
        dp2 = temp

      return dp1

    return max(rob(0, len(nums) - 2),
               rob(1, len(nums) - 1))


# Link: https://leetcode.com/problems/shifting-letters-ii/description/
class Solution:
  def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
    ans = []
    currShift = 0
    timeline = [0] * (len(s) + 1)

    for start, end, direction in shifts:
      diff = 1 if direction else -1
      timeline[start] += diff
      timeline[end + 1] -= diff

    for i, c in enumerate(s):
      currShift = (currShift + timeline[i]) % 26
      num = (ord(s[i]) - ord('a') + currShift + 26) % 26
      ans.append(chr(ord('a') + num))

    return ''.join(ans)


# Link: https://leetcode.com/problems/smallest-subarrays-with-maximum-bitwise-or/description/
class Solution:
  def smallestSubarrays(self, nums: List[int]) -> List[int]:
    kMaxBit = 30
    ans = [1] * len(nums)
    # closest[j] := the closest index i s.t. the j-th bit of nums[i] is 1
    closest = [0] * kMaxBit

    for i in reversed(range(len(nums))):
      for j in range(kMaxBit):
        if nums[i] >> j & 1:
          closest[j] = i
        ans[i] = max(ans[i], closest[j] - i + 1)

    return ans


# Link: https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/
class Solution:
  def findNumberOfLIS(self, nums: List[int]) -> int:
    ans = 0
    maxLength = 0
    # length[i] := the length of the LIS ending in nums[i]
    length = [1] * len(nums)
    # count[i] := the number of LIS's ending in nums[i]
    count = [1] * len(nums)

    # Calculate the `length` and `count` arrays.
    for i, num in enumerate(nums):
      for j in range(i):
        if nums[j] < num:
          if length[i] < length[j] + 1:
            length[i] = length[j] + 1
            count[i] = count[j]
          elif length[i] == length[j] + 1:
            count[i] += count[j]

    # Get the number of LIS.
    for i, l in enumerate(length):
      if l > maxLength:
        maxLength = l
        ans = count[i]
      elif l == maxLength:
        ans += count[i]

    return ans


# Link: https://leetcode.com/problems/perfect-squares/description/
class Solution:
  def numSquares(self, n: int) -> int:
    dp = [n] * (n + 1)  # 1^2 x n
    dp[0] = 0  # no way
    dp[1] = 1  # 1^2

    for i in range(2, n + 1):
      j = 1
      while j * j <= i:
        dp[i] = min(dp[i], dp[i - j * j] + 1)
        j += 1

    return dp[n]


# Link: https://leetcode.com/problems/frequency-tracker/description/
class FrequencyTracker:
  def __init__(self):
    self.count = collections.Counter()
    self.freqCount = collections.Counter()

  def add(self, number: int) -> None:
    if self.count[number] > 0:
      self.freqCount[self.count[number]] -= 1
    self.count[number] += 1
    self.freqCount[self.count[number]] += 1

  def deleteOne(self, number: int) -> None:
    if self.count[number] == 0:
      return
    self.freqCount[self.count[number]] -= 1
    self.count[number] -= 1
    self.freqCount[self.count[number]] += 1

  def hasFrequency(self, frequency: int) -> bool:
    return self.freqCount[frequency] > 0


# Link: https://leetcode.com/problems/ugly-number-iii/description/
class Solution:
  def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
    ab = a * b // math.gcd(a, b)
    ac = a * c // math.gcd(a, c)
    bc = b * c // math.gcd(b, c)
    abc = a * bc // math.gcd(a, bc)
    return bisect.bisect_left(
        range(2 * 10**9), n,
        key=lambda m: m // a + m // b + m // c - m // ab - m // ac - m // bc + m // abc)


# Link: https://leetcode.com/problems/find-valid-matrix-given-row-and-column-sums/description/
class Solution:
  def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
    m = len(rowSum)
    n = len(colSum)
    ans = [[0] * n for _ in range(m)]

    for i in range(m):
      for j in range(n):
        ans[i][j] = min(rowSum[i], colSum[j])
        rowSum[i] -= ans[i][j]
        colSum[j] -= ans[i][j]

    return ans


# Link: https://leetcode.com/problems/binary-tree-level-order-traversal-ii/description/
class Solution:
  def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
      return []

    ans = []
    q = collections.deque([root])

    while q:
      currLevel = []
      for _ in range(len(q)):
        node = q.popleft()
        currLevel.append(node.val)
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      ans.append(currLevel)

    return ans[::-1]


# Link: https://leetcode.com/problems/queens-that-can-attack-the-king/description/
class Solution:
  def queensAttacktheKing(self, queens: List[List[int]], king: List[int]) -> List[List[int]]:
    ans = []
    queens = {(i, j) for i, j in queens}

    for d in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
      i = king[0] + d[0]
      j = king[1] + d[1]
      while 0 <= i < 8 and 0 <= j < 8:
        if (i, j) in queens:
          ans.append([i, j])
          break
        i += d[0]
        j += d[1]

    return ans


# Link: https://leetcode.com/problems/most-popular-video-creator/description/
class Creator:
  def __init__(self, popularity: int, videoId: str, maxView: int):
    self.popularity = popularity  # the popularity sum
    self.videoId = videoId        # the video id that has the maximum view
    self.maxView = maxView        # the maximum view of the creator


class Solution:
  def mostPopularCreator(self, creators: List[str], ids: List[str], views: List[int]) -> List[List[str]]:
    ans = []
    maxPopularity = 0
    nameToCreator = {}

    for name, id, view in zip(creators, ids, views):
      if name not in nameToCreator:
        nameToCreator[name] = Creator(view, id, view)
        maxPopularity = max(maxPopularity, view)
        continue
      creator = nameToCreator[name]
      creator.popularity += view
      maxPopularity = max(maxPopularity, creator.popularity)
      if creator.maxView < view or \
              creator.maxView == view and creator.videoId > id:
        creator.videoId = id
        creator.maxView = view

    for name, creator in nameToCreator.items():
      if creator.popularity == maxPopularity:
        ans.append([name, creator.videoId])

    return ans


# Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/
class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    sell = 0
    hold = -math.inf
    prev = 0

    for price in prices:
      cache = sell
      sell = max(sell, hold + price)
      hold = max(hold, prev - price)
      prev = cache

    return sell


# Link: https://leetcode.com/problems/minimum-lines-to-represent-a-line-chart/description/
class Solution:
  def minimumLines(self, stockPrices: List[List[int]]) -> int:
    ans = 0

    stockPrices.sort()

    def getSlope(p: List[int], q: List[int]) -> Tuple[int, int]:
      dx = p[0] - q[0]
      dy = p[1] - q[1]
      if dx == 0:
        return (0, p[0])
      if dy == 0:
        return (p[1], 0)
      d = gcd(dx, dy)
      return (dx // d, dy // d)

    for i in range(2, len(stockPrices)):
      a = getSlope(stockPrices[i - 2], stockPrices[i - 1])
      b = getSlope(stockPrices[i - 1], stockPrices[i])
      if a != b:
        ans += 1

    return ans + (len(stockPrices) > 1)


# Link: https://leetcode.com/problems/non-decreasing-subsequences/description/
class Solution:
  def findSubsequences(self, nums: List[int]) -> List[List[int]]:
    ans = []

    def dfs(s: int, path: List[int]) -> None:
      if len(path) > 1:
        ans.append(path)

      used = set()

      for i in range(s, len(nums)):
        if nums[i] in used:
          continue
        if not path or nums[i] >= path[-1]:
          used.add(nums[i])
          dfs(i + 1, path + [nums[i]])

    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/prime-palindrome/description/
class Solution:
  def primePalindrome(self, n: int) -> int:
    def getPalindromes(n: int) -> int:
      length = n // 2
      for i in range(10**(length - 1), 10**length):
        s = str(i)
        for j in range(10):
          yield int(s + str(j) + s[::-1])

    def isPrime(num: int) -> bool:
      return not any(num % i == 0 for i in range(2, int(num**0.5 + 1)))

    if n <= 2:
      return 2
    if n == 3:
      return 3
    if n <= 5:
      return 5
    if n <= 7:
      return 7
    if n <= 11:
      return 11

    nLength = len(str(n))

    while True:
      for num in getPalindromes(nLength):
        if num >= n and isPrime(num):
          return num
      nLength += 1


# Link: https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock/description/
class Solution:
  def getDescentPeriods(self, prices: List[int]) -> int:
    ans = 1  # prices[0]
    dp = 1

    for i in range(1, len(prices)):
      if prices[i] == prices[i - 1] - 1:
        dp += 1
      else:
        dp = 1
      ans += dp

    return ans


# Link: https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/description/
class Solution:
  def countTriplets(self, arr: List[int]) -> int:
    ans = 0
    xors = [0]
    prefix = 0

    for i, a in enumerate(arr):
      prefix ^= a
      xors.append(prefix)

    for j in range(1, len(arr)):
      for i in range(0, j):
        xors_i = xors[j] ^ xors[i]
        for k in range(j, len(arr)):
          xors_k = xors[k + 1] ^ xors[j]
          if xors_i == xors_k:
            ans += 1

    return ans


# Link: https://leetcode.com/problems/the-number-of-weak-characters-in-the-game/description/
class Solution:
  def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
    ans = 0
    maxDefense = 0

    # Sort properties by `attack` in descending order, then by `defense` in
    # ascending order.
    for _, defense in sorted(properties, key=lambda x: (-x[0], x[1])):
      if defense < maxDefense:
        ans += 1
      maxDefense = max(maxDefense, defense)

    return ans


# Link: https://leetcode.com/problems/the-number-of-weak-characters-in-the-game/description/
class Solution:
  def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
    ans = 0
    maxAttack = max(attack for attack, _ in properties)
    # maxDefenses[i] := the maximum defense for the i-th attack
    maxDefenses = [0] * (maxAttack + 2)

    for attack, defense in properties:
      maxDefenses[attack] = max(maxDefenses[attack], defense)

    for i in range(maxAttack, 0, -1):
      maxDefenses[i] = max(maxDefenses[i], maxDefenses[i + 1])

    for attack, defense in properties:
      if maxDefenses[attack + 1] > defense:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/delete-nodes-and-return-forest/description/
class Solution:
  def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
    ans = []
    toDeleteSet = set(to_delete)

    def dfs(root: TreeNode, isRoot: bool) -> TreeNode:
      if not root:
        return None

      deleted = root.val in toDeleteSet
      if isRoot and not deleted:
        ans.append(root)

      # If root is deleted, both children have the possibility to be a new root
      root.left = dfs(root.left, deleted)
      root.right = dfs(root.right, deleted)
      return None if deleted else root

    dfs(root, True)
    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-make-a-special-number/description/
class Solution:
  def minimumOperations(self, num: str) -> int:
    n = len(num)
    seenFive = False
    seenZero = False

    for i in range(n - 1, -1, -1):
      if seenZero and num[i] == '0':  # '00'
        return n - i - 2
      if seenZero and num[i] == '5':  # '50'
        return n - i - 2
      if seenFive and num[i] == '2':  # '25'
        return n - i - 2
      if seenFive and num[i] == '7':  # '75'
        return n - i - 2
      seenZero = seenZero or num[i] == '0'
      seenFive = seenFive or num[i] == '5'

    return n - 1 if seenZero else n


# Link: https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/description/
class Solution:
  def minScore(self, n: int, roads: List[List[int]]) -> int:
    ans = math.inf
    graph = [[] for _ in range(n + 1)]  # graph[u] := [(v, distance)]
    q = collections.deque([1])
    seen = {1}

    for u, v, distance in roads:
      graph[u].append((v, distance))
      graph[v].append((u, distance))

    while q:
      u = q.popleft()
      for v, d in graph[u]:
        ans = min(ans, d)
        if v in seen:
          continue
        q.append(v)
        seen.add(v)

    return ans


# Link: https://leetcode.com/problems/making-file-names-unique/description/
class Solution:
  def getFolderNames(self, names: List[str]) -> List[str]:
    ans = []
    nameToSuffix = {}

    for name in names:
      if name in nameToSuffix:
        suffix = nameToSuffix[name] + 1
        newName = self._getName(name, suffix)
        while newName in nameToSuffix:
          suffix += 1
          newName = self._getName(name, suffix)
        nameToSuffix[name] = suffix
        nameToSuffix[newName] = 0
        ans.append(newName)
      else:
        nameToSuffix[name] = 0
        ans.append(name)

    return ans

  def _getName(self, name: str, suffix: int) -> str:
    return name + '(' + str(suffix) + ')'


# Link: https://leetcode.com/problems/make-sum-divisible-by-p/description/
class Solution:
  def minSubarray(self, nums: List[int], p: int) -> int:
    summ = sum(nums)
    remainder = summ % p
    if remainder == 0:
      return 0

    ans = len(nums)
    prefix = 0
    prefixToIndex = {0: -1}

    for i, num in enumerate(nums):
      prefix += num
      prefix %= p
      target = (prefix - remainder + p) % p
      if target in prefixToIndex:
        ans = min(ans, i - prefixToIndex[target])
      prefixToIndex[prefix] = i

    return -1 if ans == len(nums) else ans


# Link: https://leetcode.com/problems/contiguous-array/description/
class Solution:
  def findMaxLength(self, nums: List[int]) -> int:
    ans = 0
    prefix = 0
    prefixToIndex = {0: -1}

    for i, num in enumerate(nums):
      prefix += 1 if num else -1
      ans = max(ans, i - prefixToIndex.setdefault(prefix, i))

    return ans


# Link: https://leetcode.com/problems/reverse-nodes-in-even-length-groups/description/
class Solution:
  def reverseEvenLengthGroups(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # prev -> (head -> ... -> tail) -> next -> ...
    dummy = ListNode(0, head)
    prev = dummy
    tail = head
    next = head.next
    groupLength = 1

    def getTailAndLength(head: Optional[ListNode], groupLength: int) -> Tuple[Optional[ListNode], int]:
      length = 1
      tail = head
      while length < groupLength and tail.next:
        tail = tail.next
        length += 1
      return tail, length

    def reverse(head: Optional[ListNode]) -> Optional[ListNode]:
      prev = None
      while head:
        next = head.next
        head.next = prev
        prev = head
        head = next
      return prev

    while True:
      if groupLength & 1:
        prev.next = head
        prev = tail
      else:
        tail.next = None
        prev.next = reverse(head)
        # Prev -> (tail -> ... -> head) -> next -> ...
        head.next = next
        prev = head
      if not next:
        break
      head = next
      tail, length = getTailAndLength(head, groupLength + 1)
      next = tail.next
      groupLength = length

    return dummy.next


# Link: https://leetcode.com/problems/binary-search-tree-iterator-ii/description/
class BSTIterator:
  def __init__(self, root: Optional[TreeNode]):
    self.prevsAndCurr = []
    self.nexts = []
    self._pushLeftsUntilNull(root)

  def hasNext(self) -> bool:
    return len(self.nexts) > 0

  def next(self) -> int:
    root, fromNext = self.nexts.pop()
    if fromNext:
      self._pushLeftsUntilNull(root.right)
    self.prevsAndCurr.append(root)
    return root.val

  def hasPrev(self) -> bool:
    return len(self.prevsAndCurr) > 1

  def prev(self) -> int:
    self.nexts.append((self.prevsAndCurr.pop(), False))
    return self.prevsAndCurr[-1].val

  def _pushLeftsUntilNull(self, root):
    while root:
      self.nexts.append((root, True))
      root = root.left


# Link: https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
class Solution:
  def letterCombinations(self, digits: str) -> List[str]:
    if not digits:
      return []

    digitToLetters = ['', '', 'abc', 'def', 'ghi',
                      'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
    ans = []

    def dfs(i: int, path: List[chr]) -> None:
      if i == len(digits):
        ans.append(''.join(path))
        return

      for letter in digitToLetters[ord(digits[i]) - ord('0')]:
        path.append(letter)
        dfs(i + 1, path)
        path.pop()

    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
class Solution:
  def letterCombinations(self, digits: str) -> List[str]:
    if not digits:
      return []

    ans = ['']
    digitToLetters = ['', '', 'abc', 'def', 'ghi',
                      'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']

    for d in digits:
      temp = []
      for s in ans:
        for c in digitToLetters[ord(d) - ord('0')]:
          temp.append(s + c)
      ans = temp

    return ans


# Link: https://leetcode.com/problems/find-eventual-safe-states/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
    states = [State.kInit] * len(graph)

    def hasCycle(u: int) -> bool:
      if states[u] == State.kVisiting:
        return True
      if states[u] == State.kVisited:
        return False

      states[u] = State.kVisiting
      if any(hasCycle(v) for v in graph[u]):
        return True
      states[u] = State.kVisited

    return [i for i in range(len(graph)) if not hasCycle(i)]


# Link: https://leetcode.com/problems/maximum-palindromes-after-operations/description/
class Solution:
  def maxPalindromesAfterOperations(self, words: List[str]) -> int:
    ans = 0
    count = collections.Counter(''.join(words))
    pairs = sum(value // 2 for value in count.values())

    for length in sorted(len(word) for word in words):
      needPairs = length // 2
      if pairs < needPairs:
        return ans
      ans += 1
      pairs -= needPairs

    return ans


# Link: https://leetcode.com/problems/find-permutation/description/
class Solution:
  def findPermutation(self, s: str) -> List[int]:
    ans = [i for i in range(1, len(s) + 2)]

    # For each D* group (s[i..j]), reverse ans[i..j + 1].
    i = -1
    j = -1

    def getNextIndex(c: str, start: int) -> int:
      for i in range(start, len(s)):
        if s[i] == c:
          return i
      return len(s)

    while True:
      i = getNextIndex('D', j + 1)
      if i == len(s):
        break
      j = getNextIndex('I', i + 1)
      ans[i:j + 1] = ans[i:j + 1][::-1]

    return ans


# Link: https://leetcode.com/problems/find-permutation/description/
class Solution:
  def findPermutation(self, s: str) -> List[int]:
    ans = []
    stack = []

    for i, c in enumerate(s):
      stack.append(i + 1)
      if c == 'I':
        while stack:  # Consume all decreasings
          ans.append(stack.pop())
    stack.append(len(s) + 1)

    while stack:
      ans.append(stack.pop())

    return ans


# Link: https://leetcode.com/problems/minimize-maximum-of-array/description/
class Solution:
  def minimizeArrayValue(self, nums: List[int]) -> int:
    ans = 0
    prefix = 0

    for i, num in enumerate(nums):
      prefix += num
      prefixAvg = math.ceil(prefix / (i + 1))
      ans = max(ans, prefixAvg)

    return ans


# Link: https://leetcode.com/problems/convex-polygon/description/
class Solution:
  def isConvex(self, points: List[List[int]]) -> bool:
    # Pq x qr
    def getCross(p: List[int], q: List[int], r: List[int]):
      return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    sign = 0
    for i in range(len(points)):
      cross = getCross(points[i - 2], points[i - 1], points[i])
      if cross == 0:  # p, q, r are collinear.
        continue
      if sign == 0:  # Find the first cross that's not 0.
        sign = cross
      elif cross * sign < 0:
        return False

    return True


# Link: https://leetcode.com/problems/find-all-anagrams-in-a-string/description/
class Solution:
  def findAnagrams(self, s: str, p: str) -> List[int]:
    ans = []
    count = collections.Counter(p)
    required = len(p)

    for r, c in enumerate(s):
      count[c] -= 1
      if count[c] >= 0:
        required -= 1
      if r >= len(p):
        count[s[r - len(p)]] += 1
        if count[s[r - len(p)]] > 0:
          required += 1
      if required == 0:
        ans.append(r - len(p) + 1)

    return ans


# Link: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
class Solution:
  def countComponents(self, n: int, edges: List[List[int]]) -> int:
    ans = 0
    graph = [[] for _ in range(n)]
    seen = set()

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    def bfs(node: int, seen: Set[int]) -> None:
      q = collections.deque([node])
      seen.add(node)

      while q:
        u = q.pop()
        for v in graph[u]:
          if v not in seen:
            q.append(v)
            seen.add(v)

    for i in range(n):
      if i not in seen:
        bfs(i, seen)
        ans += 1

    return ans


# Link: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1
    self.count -= 1

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def countComponents(self, n: int, edges: List[List[int]]) -> int:
    uf = UnionFind(n)

    for u, v in edges:
      uf.unionByRank(u, v)

    return uf.count


# Link: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
class Solution:
  def countComponents(self, n: int, edges: List[List[int]]) -> int:
    ans = 0
    graph = [[] for _ in range(n)]
    seen = set()

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    def dfs(u: int, seen: Set[int]) -> None:
      for v in graph[u]:
        if v not in seen:
          seen.add(v)
          dfs(v, seen)

    for i in range(n):
      if i not in seen:
        seen.add(i)
        dfs(graph, i, seen)
        ans += 1

    return ans


# Link: https://leetcode.com/problems/get-equal-substrings-within-budget/description/
class Solution:
  def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
    j = 0
    for i in range(len(s)):
      maxCost -= abs(ord(s[i]) - ord(t[i]))
      if maxCost < 0:
        maxCost += abs(ord(s[j]) - ord(t[j]))
        j += 1

    return len(s) - j


# Link: https://leetcode.com/problems/minimum-cost-for-tickets/description/
class Solution:
  def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    ans = 0
    last7 = collections.deque()
    last30 = collections.deque()

    for day in days:
      while last7 and last7[0][0] + 7 <= day:
        last7.popleft()
      while last30 and last30[0][0] + 30 <= day:
        last30.popleft()
      last7.append([day, ans + costs[1]])
      last30.append([day, ans + costs[2]])
      ans = min(ans + costs[0], last7[0][1], last30[0][1])

    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-make-the-integer-zero/description/
class Solution:
  def makeTheIntegerZero(self, num1: int, num2: int) -> int:
    # If k operations are used, num1 - [(num2 + 2^{i_1}) + (num2 + 2^{i_2}) +
    # ... + (num2 + 2^{i_k})] = 0. So, num1 - k * num2 = (2^{i_1} + 2^{i_2} +
    # ... + 2^{i_k}), where i_1, i_2, ..., i_k are in the range [0, 60].
    # Note that for any number x, we can use "x's bit count" operations to make
    # x equal to 0. Additionally, we can also use x operations to deduct x by
    # 2^0 (x times), which also results in 0.

    for ops in range(61):
      target = num1 - ops * num2
      if target.bit_count() <= ops <= target:
        return ops

    return -1


# Link: https://leetcode.com/problems/maximal-range-that-each-element-is-maximum-in-it/description/
class Solution:
  def maximumLengthOfRanges(self, nums: List[int]) -> List[int]:
    ans = [0] * len(nums)
    stack = []  # a decreasing stack

    for i in range(len(nums) + 1):
      while stack and (i == len(nums) or nums[stack[-1]] < nums[i]):
        index = stack.pop()
        left = stack[-1] if stack else -1
        ans[index] = i - left - 1
      stack.append(i)

    return ans


# Link: https://leetcode.com/problems/find-the-winner-of-an-array-game/description/
class Solution:
  def getWinner(self, arr: List[int], k: int) -> int:
    ans = arr[0]
    wins = 0

    i = 1
    while i < len(arr) and wins < k:
      if arr[i] > ans:
        ans = arr[i]
        wins = 1
      else:
        wins += 1
      i += 1

    return ans


# Link: https://leetcode.com/problems/fizz-buzz-multithreaded/description/
from threading import Semaphore


class FizzBuzz:
  def __init__(self, n: int):
    self.n = n
    self.fizzSemaphore = Semaphore(0)
    self.buzzSemaphore = Semaphore(0)
    self.fizzbuzzSemaphore = Semaphore(0)
    self.numberSemaphore = Semaphore(1)

  # printFizz() outputs "fizz"
  def fizz(self, printFizz: 'Callable[[], None]') -> None:
    for i in range(1, self.n + 1):
      if i % 3 == 0 and i % 15 != 0:
        self.fizzSemaphore.acquire()
        printFizz()
        self.numberSemaphore.release()

  # printBuzz() outputs "buzz"
  def buzz(self, printBuzz: 'Callable[[], None]') -> None:
    for i in range(1, self.n + 1):
      if i % 5 == 0 and i % 15 != 0:
        self.buzzSemaphore.acquire()
        printBuzz()
        self.numberSemaphore.release()

  # printFizzBuzz() outputs "fizzbuzz"
  def fizzbuzz(self, printFizzBuzz: 'Callable[[], None]') -> None:
    for i in range(1, self.n + 1):
      if i % 15 == 0:
        self.fizzbuzzSemaphore.acquire()
        printFizzBuzz()
        self.numberSemaphore.release()

  # printNumber(x) outputs "x", where x is an integer.
  def number(self, printNumber: 'Callable[[int], None]') -> None:
    for i in range(1, self.n + 1):
      self.numberSemaphore.acquire()
      if i % 15 == 0:
        self.fizzbuzzSemaphore.release()
      elif i % 3 == 0:
        self.fizzSemaphore.release()
      elif i % 5 == 0:
        self.buzzSemaphore.release()
      else:
        printNumber(i)
        self.numberSemaphore.release()


# Link: https://leetcode.com/problems/find-smallest-common-element-in-all-rows/description/
class Solution:
  def smallestCommonElement(self, mat: List[List[int]]) -> int:
    kMax = 10000
    count = [0] * (kMax + 1)

    for row in mat:
      for a in row:
        count[a] += 1
        if count[a] == len(mat):
          return a

    return -1


# Link: https://leetcode.com/problems/complete-binary-tree-inserter/description/
class CBTInserter:
  def __init__(self, root: Optional[TreeNode]):
    self.tree = [root]
    for node in self.tree:
      if node.left:
        self.tree.append(node.left)
      if node.right:
        self.tree.append(node.right)

  def insert(self, v: int) -> int:
    n = len(self.tree)
    self.tree.append(TreeNode(v))
    parent = self.tree[(n - 1) // 2]
    if n & 1:
      parent.left = self.tree[-1]
    else:
      parent.right = self.tree[-1]
    return parent.val

  def get_root(self) -> Optional[TreeNode]:
    return self.tree[0]


# Link: https://leetcode.com/problems/furthest-building-you-can-reach/description/
class Solution:
  def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
    minHeap = []

    for i, (a, b) in enumerate(itertools.pairwise(heights)):
      diff = b - a
      if diff <= 0:
        continue
      heapq.heappush(minHeap, diff)
      # If we run out of ladders, greedily use as less bricks as possible.
      if len(minHeap) > ladders:
        bricks -= heapq.heappop(minHeap)
      if bricks < 0:
        return i

    return len(heights) - 1


# Link: https://leetcode.com/problems/additive-number/description/
class Solution:
  def isAdditiveNumber(self, num: str) -> bool:
    n = len(num)

    def dfs(firstNum: int, secondNum: int, s: int) -> bool:
      if s == len(num):
        return True

      thirdNum = firstNum + secondNum
      thirdNumStr = str(thirdNum)

      return num.find(thirdNumStr, s) == s and dfs(secondNum, thirdNum, s + len(thirdNumStr))

    # num[0..i] = firstNum
    for i in range(n // 2):
      if i > 0 and num[0] == '0':
        return False
      firstNum = int(num[:i + 1])
      # num[i + 1..j] = secondNum
      # |thirdNum| >= max(|firstNum|, |secondNum|)
      j = i + 1
      while max(i, j - i) < n - j:
        if j > i + 1 and num[i + 1] == '0':
          break
        secondNum = int(num[i + 1:j + 1])
        if dfs(firstNum, secondNum, j + 1):
          return True
        j += 1

    return False


# Link: https://leetcode.com/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list/description/
class Solution:
  def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
    ans = []
    n = len(favoriteCompanies)
    companies = [set(comp) for comp in favoriteCompanies]

    for i in range(n):
      find = False
      for j in range(n):
        if i == j:
          continue
        if companies[i].issubset(companies[j]):
          find = True
          break
      if not find:
        ans.append(i)

    return ans


# Link: https://leetcode.com/problems/the-time-when-the-network-becomes-idle/description/
class Solution:
  def networkBecomesIdle(self, edges: List[List[int]], patience: List[int]) -> int:
    n = len(patience)
    ans = 0
    graph = [[] for _ in range(n)]
    q = collections.deque([0])
    dist = [math.inf] * n  # dist[i] := the distance between i and 0
    dist[0] = 0

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    while q:
      for _ in range(len(q)):
        u = q.popleft()
        for v in graph[u]:
          if dist[v] == math.inf:
            dist[v] = dist[u] + 1
            q.append(v)

    for i in range(1, n):
      numResending = (dist[i] * 2 - 1) // patience[i]
      lastResendingTime = patience[i] * numResending
      lastArrivingTime = lastResendingTime + dist[i] * 2
      ans = max(ans, lastArrivingTime)

    return ans + 1


# Link: https://leetcode.com/problems/form-array-by-concatenating-subarrays-of-another-array/description/
class Solution:
  def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
    i = 0  # groups' index
    j = 0  # nums' index

    while i < len(groups) and j < len(nums):
      if self._isMatch(groups[i], nums, j):
        j += len(groups[i])
        i += 1
      else:
        j += 1

    return i == len(groups)

  # Returns True if group == nums[j..j + |group|].
  def _isMatch(self, group: List[int], nums: List[int], j: int) -> bool:
    if j + |group| > len(nums):
      return False
    for i, g in enumerate(group):
      if g != nums[j + i]:
        return False
    return True


# Link: https://leetcode.com/problems/letter-tile-possibilities/description/
class Solution:
  def numTilePossibilities(self, tiles: str) -> int:
    count = collections.Counter(tiles)

    def dfs(count: Dict[int, int]) -> int:
      possibleSequences = 0

      for k, v in count.items():
        if v == 0:
          continue
        # Put c in the current position. We only care about the number of possible
        # sequences of letters but don't care about the actual combination.
        count[k] -= 1
        possibleSequences += 1 + dfs(count)
        count[k] += 1

      return possibleSequences

    return dfs(count)


# Link: https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/description/
class Solution:
  def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
    ans = 0
    distinct = 0
    count = [0] * 128

    l = 0
    for r, c in enumerate(s):
      count[ord(c)] += 1
      if count[ord(c)] == 1:
        distinct += 1
      while distinct == 3:
        count[ord(s[l])] -= 1
        if count[ord(s[l])] == 0:
          distinct -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/word-pattern-ii/description/
class Solution:
  def wordPatternMatch(self, pattern: str, s: str) -> bool:
    def isMatch(i: int, j: int, charToString: Dict[chr, str], seen: Set[str]) -> bool:
      if i == len(pattern) and j == len(s):
        return True
      if i == len(pattern) or j == len(s):
        return False

      c = pattern[i]

      if c in charToString:
        t = charToString[c]
        # See if we can match t with s[j..n).
        if t not in s[j:]:
          return False

        # If there's a match, continue to match the rest.
        return isMatch(i + 1, j + len(t), charToString, seen)

      for k in range(j, len(s)):
        t = s[j:k + 1]

        # This string is mapped by another character.
        if t in seen:
          continue

        charToString[c] = t
        seen.add(t)

        if isMatch(i + 1, k + 1, charToString, seen):
          return True

        # Backtrack.
        del charToString[c]
        seen.remove(t)

      return False

    return isMatch(0, 0, {}, set())


# Link: https://leetcode.com/problems/add-bold-tag-in-string/description/
class Solution:
  def addBoldTag(self, s: str, words: List[str]) -> str:
    n = len(s)
    ans = []
    # bold[i] := True if s[i] should be bolded
    bold = [0] * n

    boldEnd = -1  # s[i:boldEnd] should be bolded
    for i in range(n):
      for word in words:
        if s[i:].startswith(word):
          boldEnd = max(boldEnd, i + len(word))
      bold[i] = boldEnd > i

    # Construct the with bold tags
    i = 0
    while i < n:
      if bold[i]:
        j = i
        while j < n and bold[j]:
          j += 1
        # `s[i..j)` should be bolded.
        ans.append('<b>' + s[i:j] + '</b>')
        i = j
      else:
        ans.append(s[i])
        i += 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/add-bold-tag-in-string/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.isWord = False


class Solution:
  def addBoldTag(self, s: str, words: List[str]) -> str:
    n = len(s)
    ans = []
    # bold[i] := True if s[i] should be bolded
    bold = [0] * n
    root = TrieNode()

    def insert(word: str) -> None:
      node = root
      for c in word:
        node = node.children.setdefault(c, TrieNode())
      node.isWord = True

    def find(s: str, i: int) -> int:
      node = root
      ans = -1
      for j in range(i, len(s)):
        if s[j] not in node.children:
          node.children[s[j]] = TrieNode()
        node = node.children[s[j]]
        if node.isWord:
          ans = j
      return ans

    for word in words:
      insert(word)

    boldEnd = -1  # `s[i..boldEnd]` should be bolded.
    for i in range(n):
      boldEnd = max(boldEnd, find(s, i))
      bold[i] = boldEnd >= i

    # Construct the with bold tags
    i = 0
    while i < n:
      if bold[i]:
        j = i
        while j < n and bold[j]:
          j += 1
        # `s[i..j)` should be bolded.
        ans.append('<b>' + s[i:j] + '</b>')
        i = j
      else:
        ans.append(s[i])
        i += 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/container-with-most-water/description/
class Solution:
  def maxArea(self, height: List[int]) -> int:
    ans = 0
    l = 0
    r = len(height) - 1

    while l < r:
      minHeight = min(height[l], height[r])
      ans = max(ans, minHeight * (r - l))
      if height[l] < height[r]:
        l += 1
      else:
        r -= 1

    return ans


# Link: https://leetcode.com/problems/linked-list-frequency/description/
class Solution:
  def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
    count = collections.Counter()
    curr = head

    while curr:
      count[curr.val] += 1
      curr = curr.next

    dummy = ListNode(0)
    tail = dummy

    for freq in count.values():
      tail.next = ListNode(freq)
      tail = tail.next

    return dummy.next


# Link: https://leetcode.com/problems/where-will-the-ball-fall/description/
class Solution:
  def findBall(self, grid: List[List[int]]) -> List[int]:
    m = len(grid)
    n = len(grid[0])
    # dp[i] := status of the i-th column
    # -1 := empty, 0 := b0, 1 := b1, ...
    dp = [i for i in range(n)]
    # ans[i] := the i-th ball's final positio
    ans = [-1] * n

    for i in range(m):
      newDp = [-1] * n
      for j in range(n):
        # out-of-bounds
        if j + grid[i][j] < 0 or j + grid[i][j] == n:
          continue
        if grid[i][j] == 1 and grid[i][j + 1] == -1 or \
                grid[i][j] == -1 and grid[i][j - 1] == 1:
          continue
        newDp[j + grid[i][j]] = dp[j]
      dp = newDp

    for i, ball in enumerate(dp):
      if ball != -1:
        ans[ball] = i

    return ans


# Link: https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/description/
class Solution:
  def countRestrictedPaths(self, n: int, edges: List[List[int]]) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
      graph[u - 1].append((v - 1, w))
      graph[v - 1].append((u - 1, w))

    return self._dijkstra(graph, 0, n - 1)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, dst: int) -> int:
    kMod = 10**9 + 7
    # ways[i] := the number of restricted path from i to n
    ways = [0] * len(graph)
    # dist[i] := the distance to the last node of i
    dist = [math.inf] * len(graph)

    ways[dst] = 1
    dist[dst] = 0
    minHeap = [(dist[dst], dst)]  # (d, u)

    while minHeap:
      d, u = heapq.heappop(minHeap)
      if d > dist[u]:
        continue
      for v, w in graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (dist[v], v))
        if dist[v] < dist[u]:
          ways[u] += ways[v]
          ways[u] %= kMod

    return ways[src]


# Link: https://leetcode.com/problems/minimum-moves-to-equal-array-elements/description/
class Solution:
  def minMoves(self, nums: List[int]) -> int:
    mini = min(nums)
    return sum(num - mini for num in nums)


# Link: https://leetcode.com/problems/minimum-operations-to-write-the-letter-y-on-a-grid/description/
class Solution:
  def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
    n = len(grid)
    mid = n // 2

    def getOperations(a: int, b: int) -> int:
      """Returns the number of operations to turn Y into a and non-Y into b."""
      operations = 0
      for i, row in enumerate(grid):
        for j, num in enumerate(row):
          # For the 'Y' pattern, before the midpoint, check the diagonal and
          # anti-diagonal. After the midpoint, check the middle column.
          if (i < mid and (i == j or i + j == n - 1)) or i >= mid and j == mid:
            if num != a:
              operations += 1
          elif num != b:
            operations += 1
      return operations

    return min(getOperations(0, 1), getOperations(0, 2),
               getOperations(1, 0), getOperations(1, 2),
               getOperations(2, 0), getOperations(2, 1))


# Link: https://leetcode.com/problems/minimize-length-of-array-using-operations/description/
class Solution:
  def minimumArrayLength(self, nums: List[int]) -> int:
    # Let the minimum number in the array `nums` be x.
    # * If there exists any element nums[i] where nums[i] % x > 0, a new
    #   minimum can be generated and all other numbers can be removed.
    # * If not, count the frequency of x in `nums`. For each pair of x, a 0 is
    #   generated which cannot be removed. Therefore, the result will be
    #   (frequency of x + 1) / 2.
    minNum = min(nums)
    if any(num % minNum > 0 for num in nums):
      return 1
    return (nums.count(minNum) + 1) // 2


# Link: https://leetcode.com/problems/maximum-number-of-accepted-invitations/description/
class Solution:
  def maximumInvitations(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    ans = 0
    mates = [-1] * n  # mates[i] := the i-th girl's mate

    def canInvite(i: int, seen: List[bool]) -> bool:
      """Returns True if the i-th boy can make an invitation."""
      # The i-th boy asks each girl.
      for j in range(n):
        if not grid[i][j] or seen[j]:
          continue
        seen[j] = True
        if mates[j] == -1 or canInvite(mates[j], seen):
          mates[j] = i  # Match the j-th girl with i-th boy.
          return True
      return False

    for i in range(m):
      seen = [False] * n
      if canInvite(i, seen):
        ans += 1

    return ans


# Link: https://leetcode.com/problems/robot-bounded-in-circle/description/
class Solution:
  def isRobotBounded(self, instructions: str) -> bool:
    x = 0
    y = 0
    d = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for instruction in instructions:
      if instruction == 'G':
        x += directions[d][0]
        y += directions[d][1]
      elif instruction == 'L':
        d = (d + 3) % 4
      else:
        d = (d + 1) % 4

    return (x, y) == (0, 0) or d > 0


# Link: https://leetcode.com/problems/set-matrix-zeroes/description/
class Solution:
  def setZeroes(self, matrix: List[List[int]]) -> None:
    m = len(matrix)
    n = len(matrix[0])
    shouldFillFirstRow = 0 in matrix[0]
    shouldFillFirstCol = 0 in list(zip(*matrix))[0]

    # Store the information in the first row and the first column.
    for i in range(1, m):
      for j in range(1, n):
        if matrix[i][j] == 0:
          matrix[i][0] = 0
          matrix[0][j] = 0

    # Fill 0s for the matrix except the first row and the first column.
    for i in range(1, m):
      for j in range(1, n):
        if matrix[i][0] == 0 or matrix[0][j] == 0:
          matrix[i][j] = 0

    # Fill 0s for the first row if needed.
    if shouldFillFirstRow:
      matrix[0] = [0] * n

    # Fill 0s for the first column if needed.
    if shouldFillFirstCol:
      for row in matrix:
        row[0] = 0


# Link: https://leetcode.com/problems/minimum-amount-of-time-to-collect-garbage/description/
class Solution:
  def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
    prefix = list(itertools.accumulate(travel))

    def getTime(c: str) -> int:
      characterCount = 0
      lastIndex = -1
      for i, s in enumerate(garbage):
        if any(g == c for g in s):
          lastIndex = i
        characterCount += s.count(c)
      return characterCount + (0 if lastIndex <= 0 else prefix[lastIndex - 1])

    return getTime('M') + getTime('P') + getTime('G')


# Link: https://leetcode.com/problems/sentence-screen-fitting/description/
class Solution:
  def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
    combined = ' '.join(sentence) + ' '
    n = len(combined)
    i = 0

    for _ in range(rows):
      i += cols
      if combined[i % n] == ' ':
        i += 1
      else:
        while i > 0 and combined[(i - 1) % n] != ' ':
          i -= 1

    return i // n


# Link: https://leetcode.com/problems/ways-to-split-array-into-good-subarrays/description/
class Solution:
  def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
    if 1 not in nums:
      return 0

    kMod = 1_000_000_007
    prev = -1  # the previous index of 1
    ans = 1

    for i, num in enumerate(nums):
      if num == 1:
        if prev != -1:
          ans *= i - prev
          ans %= kMod
        prev = i

    return ans


# Link: https://leetcode.com/problems/insert-interval/description/
class Solution:
  def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    n = len(intervals)
    ans = []
    i = 0

    while i < n and intervals[i][1] < newInterval[0]:
      ans.append(intervals[i])
      i += 1

    # Merge overlapping intervals.
    while i < n and intervals[i][0] <= newInterval[1]:
      newInterval[0] = min(newInterval[0], intervals[i][0])
      newInterval[1] = max(newInterval[1], intervals[i][1])
      i += 1

    ans.append(newInterval)

    while i < n:
      ans.append(intervals[i])
      i += 1

    return ans


# Link: https://leetcode.com/problems/count-primes/description/
class Solution:
  def countPrimes(self, n: int) -> int:
    if n <= 2:
      return 0
    return sum(self._sieveEratosthenes(n))

  def _sieveEratosthenes(self, n: int) -> List[bool]:
    isPrime = [True] * n
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n**0.5) + 1):
      if isPrime[i]:
        for j in range(i * i, n, i):
          isPrime[j] = False
    return isPrime


# Link: https://leetcode.com/problems/diameter-of-n-ary-tree/description/
class Solution:
  def diameter(self, root: 'Node') -> int:
    ans = 0

    def maxDepth(root: 'Node') -> int:
      nonlocal ans
      max1 = 0
      max2 = 0

      for child in root.children:
        depth = maxDepth(child)
        if depth > max1:
          max2 = max1
          max1 = depth
        elif depth > max2:
          max2 = depth

      ans = max(ans, max1 + max2)
      return 1 + max1

    maxDepth(root)
    return ans


# Link: https://leetcode.com/problems/maximum-sum-of-an-hourglass/description/
class Solution:
  def maxSum(self, grid: List[List[int]]) -> int:
    return max(grid[i - 1][j - 1] + grid[i - 1][j] + grid[i - 1][j + 1] + grid[i][j] +
               grid[i + 1][j - 1] + grid[i + 1][j] + grid[i + 1][j + 1]
               for i in range(1, len(grid) - 1)
               for j in range(1, len(grid[0]) - 1))


# Link: https://leetcode.com/problems/find-the-winner-of-the-circular-game/description/
class Solution:
  def findTheWinner(self, n: int, k: int) -> int:
    # e.g. n = 4, k = 2.
    # By using 0-indexed notation, we have the following circle:
    #
    # 0 -> 1 -> 2 -> 3 -> 0
    #      x
    #           0 -> 1 -> 2 -> 0
    #
    # After the first round, 1 is removed.
    # So, 2 becomes 0, 3 becomes 1, and 0 becomes 2.
    # Let's denote that oldIndex = f(n, k) and newIndex = f(n - 1, k).
    # By observation, we know f(n, k) = (f(n - 1, k) + k) % n.
    def f(n: int, k: int) -> int:
      if n == 1:
        return 0
      return (f(n - 1, k) + k) % n

    # Converts back to 1-indexed.
    return f(n, k) + 1


# Link: https://leetcode.com/problems/find-the-winner-of-the-circular-game/description/
class Solution:
  def findTheWinner(self, n: int, k: int) -> int:
    # e.g. n = 4, k = 2.
    # By using 0-indexed notation, we have the following circle:
    #
    # 0 -> 1 -> 2 -> 3 -> 0
    #      x
    #           0 -> 1 -> 2 -> 0
    #
    # After the first round, 1 is removed.
    # So, 2 becomes 0, 3 becomes 1, and 0 becomes 2.
    # Let's denote that oldIndex = f(n, k) and newIndex = f(n - 1, k).
    # By observation, we know f(n, k) = (f(n - 1, k) + k) % n.
    def f(n: int, k: int) -> int:
      ans = 0  # f(1, k)
      # Computes f(i, k) based on f(i - 1, k).
      for i in range(2, n + 1):
        ans = (ans + k) % i
      return ans

    # Converts back to 1-indexed.
    return f(n, k) + 1


# Link: https://leetcode.com/problems/find-the-winner-of-the-circular-game/description/
class Solution:
  def findTheWinner(self, n: int, k: int) -> int:
    # True if i-th friend is left
    friends = [False] * n

    friendCount = n
    fp = 0  # friends' index

    while friendCount > 1:
      for _ in range(k):
        while friends[fp % n]:  # The friend is not there.
          fp += 1  # Point to the next one.
        fp += 1
      friends[(fp - 1) % n] = True
      friendCount -= 1

    fp = 0
    while friends[fp]:
      fp += 1

    return fp + 1


# Link: https://leetcode.com/problems/minimum-absolute-difference-between-elements-with-constraint/description/
from sortedcontainers import SortedSet


class Solution:
  def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
    ans = math.inf
    seen = SortedSet()

    for i in range(x, len(nums)):
      seen.add(nums[i - x])
      it = seen.bisect_left(nums[i])
      if it != len(seen):
        ans = min(ans, seen[it] - nums[i])
      if it != 0:
        ans = min(ans, nums[i] - seen[it - 1])

    return ans


# Link: https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/description/
class Solution:
  def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    ans = [set() for _ in range(n)]
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n

    # Build the graph.
    for u, v in edges:
      graph[u].append(v)
      inDegrees[v] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    while q:
      for _ in range(len(q)):
        u = q.popleft()
        for v in graph[u]:
          ans[v].add(u)
          ans[v].update(ans[u])
          inDegrees[v] -= 1
          if inDegrees[v] == 0:
            q.append(v)

    return [sorted(nodes) for nodes in ans]


# Link: https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/description/
class Solution:
  def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    ans = [[] for _ in range(n)]
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)

    def dfs(u: int, ancestor: int, seen: Set[int]) -> None:
      seen.add(u)
      for v in graph[u]:
        if v in seen:
          continue
        ans[v].append(ancestor)
        dfs(v, ancestor, seen)

    for i in range(n):
      dfs(i, i, set())

    return ans


# Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    sell = 0
    hold = -math.inf

    for price in prices:
      sell = max(sell, hold + price)
      hold = max(hold, sell - price)

    return sell


# Link: https://leetcode.com/problems/longest-mountain-in-array/description/
class Solution:
  def longestMountain(self, arr: List[int]) -> int:
    ans = 0
    i = 0

    while i + 1 < len(arr):
      while i + 1 < len(arr) and arr[i] == arr[i + 1]:
        i += 1

      increasing = 0
      decreasing = 0

      while i + 1 < len(arr) and arr[i] < arr[i + 1]:
        increasing += 1
        i += 1

      while i + 1 < len(arr) and arr[i] > arr[i + 1]:
        decreasing += 1
        i += 1

      if increasing > 0 and decreasing > 0:
        ans = max(ans, increasing + decreasing + 1)

    return ans


# Link: https://leetcode.com/problems/path-sum-iv/description/
class Solution:
  def pathSum(self, nums: List[int]) -> int:
    ans = 0
    tree = [[-1] * 8 for _ in range(4)]

    for num in nums:
      d = num // 100 - 1
      p = (num % 100) // 10 - 1
      v = num % 10
      tree[d][p] = v

    def dfs(i: int, j: int, path: int) -> None:
      nonlocal ans
      if tree[i][j] == -1:
        return
      if i == 3 or max(tree[i + 1][j * 2], tree[i + 1][j * 2 + 1]) == -1:
        ans += path + tree[i][j]
        return

      dfs(i + 1, j * 2, path + tree[i][j])
      dfs(i + 1, j * 2 + 1, path + tree[i][j])

    dfs(0, 0, 0)
    return ans


# Link: https://leetcode.com/problems/valid-palindrome-iv/description/
class Solution:
  def makePalindrome(self, s: str) -> bool:
    change = 0
    l = 0
    r = len(s) - 1

    while l < r:
      if s[l] != s[r]:
        change += 1
        if change > 2:
          return False
      l += 1
      r -= 1

    return True


# Link: https://leetcode.com/problems/number-of-matching-subsequences/description/
class Solution:
  def numMatchingSubseq(self, s: str, words: List[str]) -> int:
    ans = 0
    # [(i, j)] := words[i] and the letter words[i][j] is waiting for
    bucket = [[] for _ in range(26)]

    # For each word, it's waiting for word[0].
    for i, word in enumerate(words):
      bucket[ord(word[0]) - ord('a')].append((i, 0))

    for c in s:
      # Let prevBucket = bucket[c] and clear bucket[c].
      index = ord(c) - ord('a')
      prevBucket = bucket[index]
      bucket[index] = []
      for i, j in prevBucket:
        j += 1
        if j == len(words[i]):  # All the letters in words[i] are matched.
          ans += 1
        else:
          bucket[ord(words[i][j]) - ord('a')].append((i, j))

    return ans


# Link: https://leetcode.com/problems/number-of-matching-subsequences/description/
class Solution:
  def numMatchingSubseq(self, s: str, words: List[str]) -> int:
    root = {}

    def insert(word: str) -> None:
      node = root
      for c in word:
        if c not in node:
          node[c] = {'count': 0}
        node = node[c]
      node['count'] += 1

    for word in words:
      insert(word)

    def dfs(s: str, i: int, node: dict) -> int:
      ans = node['count'] if 'count' in node else 0

      if i >= len(s):
        return ans

      for c in string.ascii_lowercase:
        if c in node:
          try:
            index = s.index(c, i)
            ans += dfs(s, index + 1, node[c])
          except ValueError:
            continue

      return ans

    return dfs(s, 0, root)


# Link: https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/description/
class Solution:
  def minOperations(self, nums: List[int], x: int) -> int:
    targetSum = sum(nums) - x
    if targetSum == 0:
      return len(nums)
    maxLen = self._maxSubArrayLen(nums, targetSum)
    return -1 if maxLen == -1 else len(nums) - maxLen

  # Same as 325. Maximum Size Subarray Sum Equals k
  def _maxSubArrayLen(self, nums: List[int], k: int) -> int:
    res = -1
    prefix = 0
    prefixToIndex = {0: -1}

    for i, num in enumerate(nums):
      prefix += num
      target = prefix - k
      if target in prefixToIndex:
        res = max(res, i - prefixToIndex[target])
      # No need to check the existence of the prefix since it's unique.
      prefixToIndex[prefix] = i

    return res


# Link: https://leetcode.com/problems/design-tic-tac-toe/description/
class TicTacToe:
  def __init__(self, n: int):
    self.n = n
    # Record count('X') - count('O').
    self.rows = [0] * n
    self.cols = [0] * n
    self.diag = 0
    self.antiDiag = 0

  """ Player {player} makes a move at ({row}, {col}).

      @param row    The row of the board.
      @param col    The column of the board.
      @param player The player, can be either 1 or 2.
      @return The current winning condition, can be either:
              0: No one wins.
              1: Player 1 wins.
              2: Player 2 wins.
  """

  def move(self, row: int, col: int, player: int) -> int:
    toAdd = 1 if player == 1 else -1
    target = self.n if player == 1 else -self.n

    if row == col:
      self.diag += toAdd
      if self.diag == target:
        return player

    if row + col == self.n - 1:
      self.antiDiag += toAdd
      if self.antiDiag == target:
        return player

    self.rows[row] += toAdd
    if self.rows[row] == target:
      return player

    self.cols[col] += toAdd
    if self.cols[col] == target:
      return player

    return 0


# Link: https://leetcode.com/problems/apply-discount-every-n-orders/description/
class Cashier:
  def __init__(self, n: int, discount: int, products: List[int], prices: List[int]):
    self.n = n
    self.discount = discount
    self.productToPrice = dict(zip(products, prices))
    self.count = 0

  def getBill(self, product: List[int], amount: List[int]) -> float:
    self.count += 1
    total = sum(self.productToPrice[p] * amount[i]
                for i, p in enumerate(product))
    if self.count % self.n == 0:
      return total * (1 - self.discount / 100)
    return total


# Link: https://leetcode.com/problems/find-beautiful-indices-in-the-given-array-i/description/
class Solution:
  def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
    ans = []
    indicesA = self._kmp(s, a)
    indicesB = self._kmp(s, b)
    indicesBIndex = 0  # indicesB' index

    for i in indicesA:
      # The constraint is: |j - i| <= k. So, -k <= j - i <= k. So, move
      # `indicesBIndex` s.t. j - i >= -k, where j := indicesB[indicesBIndex].
      while indicesBIndex < len(indicesB) and indicesB[indicesBIndex] - i < -k:
        indicesBIndex += 1
      if indicesBIndex < len(indicesB) and indicesB[indicesBIndex] - i <= k:
        ans.append(i)

    return ans

  def _kmp(self, s: str, pattern: str) -> List[int]:
    """Returns the starting indices of all occurrences of the pattern in `s`."""

    def getLPS(pattern: str) -> List[int]:
      """
      Returns the lps array, where lps[i] is the length of the longest prefix of
      pattern[0..i] which is also a suffix of this substring.
      """
      lps = [0] * len(pattern)
      j = 0
      for i in range(1, len(pattern)):
        while j > 0 and pattern[j] != pattern[i]:
          j = lps[j - 1]
        if pattern[i] == pattern[j]:
          lps[i] = j + 1
          j += 1
      return lps

    res = []
    lps = getLPS(pattern)
    i = 0  # s' index
    j = 0  # pattern's index
    while i < len(s):
      if s[i] == pattern[j]:
        i += 1
        j += 1
        if j == len(pattern):
          res.append(i - j)
          j = lps[j - 1]
      # Mismatch after j matches.
      elif j != 0:
          # Don't match lps[0..lps[j - 1]] since they will match anyway.
        j = lps[j - 1]
      else:
        i += 1
    return res


# Link: https://leetcode.com/problems/majority-element-ii/description/
class Solution:
  def majorityElement(self, nums: List[int]) -> List[int]:
    ans1 = 0
    ans2 = 1
    count1 = 0
    count2 = 0

    for num in nums:
      if num == ans1:
        count1 += 1
      elif num == ans2:
        count2 += 1
      elif count1 == 0:
        ans1 = num
        count1 = 1
      elif count2 == 0:
        ans2 = num
        count2 = 1
      else:
        count1 -= 1
        count2 -= 1

    return [ans for ans in (ans1, ans2) if nums.count(ans) > len(nums) // 3]


# Link: https://leetcode.com/problems/largest-bst-subtree/description/
from dataclasses import dataclass


@dataclass(frozen=True)
class T:
  min: int  # the minimum value in the subtree
  max: int  # the maximum value in the subtree
  size: int  # the size of the subtree


class Solution:
  def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
    def dfs(root: Optional[TreeNode]) -> T:
      if not root:
        return T(math.inf, -math.inf, 0)

      l = dfs(root.left)
      r = dfs(root.right)

      if l.max < root.val < r.min:
        return T(min(l.min, root.val), max(r.max, root.val), 1 + l.size + r.size)

      # Mark one as invalid, but still record the size of children.
      # Return (-inf, inf) because no node will be > inf or < -inf.
      return T(-math.inf, math.inf, max(l.size, r.size))

    return dfs(root).size


# Link: https://leetcode.com/problems/lexicographically-smallest-string-after-substring-operation/description/
class Solution:
  def smallestString(self, s: str) -> str:
    chars = list(s)
    n = len(s)
    i = 0

    while i < n and chars[i] == 'a':
      i += 1
    if i == n:
      chars[-1] = 'z'
      return ''.join(chars)

    while i < n and s[i] != 'a':
      chars[i] = chr(ord(chars[i]) - 1)
      i += 1

    return ''.join(chars)


# Link: https://leetcode.com/problems/maximum-split-of-positive-even-integers/description/
class Solution:
  def maximumEvenSplit(self, finalSum: int) -> List[int]:
    if finalSum & 1:
      return []

    ans = []
    needSum = finalSum
    even = 2

    while needSum - even >= even + 2:
      ans.append(even)
      needSum -= even
      even += 2

    return ans + [needSum]


# Link: https://leetcode.com/problems/array-nesting/description/
class Solution:
  def arrayNesting(self, nums: List[int]) -> int:
    ans = 0

    for num in nums:
      if num == -1:
        continue
      index = num
      count = 0
      while nums[index] != -1:
        cache = index
        index = nums[index]
        nums[cache] = -1
        count += 1
      ans = max(ans, count)

    return ans


# Link: https://leetcode.com/problems/single-threaded-cpu/description/
class Solution:
  def getOrder(self, tasks: List[List[int]]) -> List[int]:
    n = len(tasks)
    A = [[*task, i] for i, task in enumerate(tasks)]
    ans = []
    minHeap = []
    i = 0  # tasks' index
    time = 0  # the current time

    A.sort()

    while i < n or minHeap:
      if not minHeap:
        time = max(time, A[i][0])
      while i < n and time >= A[i][0]:
        heapq.heappush(minHeap, (A[i][1], A[i][2]))
        i += 1
      procTime, index = heapq.heappop(minHeap)
      time += procTime
      ans.append(index)

    return ans


# Link: https://leetcode.com/problems/fair-distribution-of-cookies/description/
class Solution:
  def distributeCookies(self, cookies: List[int], k: int) -> int:
    ans = math.inf

    def dfs(s: int, children: List[int]) -> None:
      nonlocal ans
      if s == len(cookies):
        ans = min(ans, max(children))
        return

      for i in range(k):
        children[i] += cookies[s]
        dfs(s + 1, children)
        children[i] -= cookies[s]

    dfs(0, [0] * k)
    return ans


# Link: https://leetcode.com/problems/check-if-word-is-valid-after-substitutions/description/
class Solution:
  def isValid(self, s: str) -> bool:
    stack = []

    for c in s:
      if c == 'c':
        if len(stack) < 2 or stack[-2] != 'a' or stack[-1] != 'b':
          return False
        stack.pop()
        stack.pop()
      else:
        stack.append(c)

    return not stack


# Link: https://leetcode.com/problems/reward-top-k-students/description/
class Solution:
  def topStudents(self, positive_feedback: List[str], negative_feedback: List[str], report: List[str], student_id: List[int], k: int) -> List[int]:
    scoreAndIds = []
    pos = set(positive_feedback)
    neg = set(negative_feedback)

    for sid, r in zip(student_id, report):
      score = 0
      for word in r.split():
        if word in pos:
          score += 3
        if word in neg:
          score -= 1
      scoreAndIds.append((-score, sid))

    return [sid for _, sid in sorted(scoreAndIds)[:k]]


# Link: https://leetcode.com/problems/count-the-number-of-complete-components/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n
    self.nodeCount = [1] * n
    self.edgeCount = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self.find(u)
    j = self.find(v)
    self.edgeCount[i] += 1
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
      self.edgeCount[j] += self.edgeCount[i]
      self.nodeCount[j] += self.nodeCount[i]
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
      self.edgeCount[i] += self.edgeCount[j]
      self.nodeCount[i] += self.nodeCount[j]
    else:
      self.id[i] = j
      self.edgeCount[j] += self.edgeCount[i]
      self.nodeCount[j] += self.nodeCount[i]
      self.rank[j] += 1

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]

  def isComplete(self, u):
    return self.nodeCount[u] * (self.nodeCount[u] - 1) // 2 == self.edgeCount[u]


class Solution:
  def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
    ans = 0
    uf = UnionFind(n)
    parents = set()

    for u, v in edges:
      uf.unionByRank(u, v)

    for i in range(n):
      parent = uf.find(i)
      if parent not in parents and uf.isComplete(parent):
        ans += 1
        parents.add(parent)

    return ans


# Link: https://leetcode.com/problems/map-sum-pairs/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.sum = 0


class MapSum:
  def __init__(self):
    self.root = TrieNode()
    self.keyToVal = {}

  def insert(self, key: str, val: int) -> None:
    diff = val - self.keyToVal.get(key, 0)
    node: TrieNode = self.root
    for c in key:
      node = node.children.setdefault(c, TrieNode())
      node.sum += diff
    self.keyToVal[key] = val

  def sum(self, prefix: str) -> int:
    node: TrieNode = self.root
    for c in prefix:
      if c not in node.children:
        return 0
      node = node.children[c]
    return node.sum


# Link: https://leetcode.com/problems/break-a-palindrome/description/
class Solution:
  def breakPalindrome(self, palindrome: str) -> str:
    if len(palindrome) == 1:
      return ''

    ans = list(palindrome)

    for i in range(len(palindrome) // 2):
      if palindrome[i] != 'a':
        ans[i] = 'a'
        return ''.join(ans)

    ans[-1] = 'b'
    return ''.join(ans)


# Link: https://leetcode.com/problems/determine-the-minimum-sum-of-a-k-avoiding-array/description/
class Solution:
  def minimumSum(self, n: int, k: int) -> int:
    # These are the unique pairs that sum up to k:
    # (1, k - 1), (2, k - 2), ..., (ceil(k // 2), floor(k // 2)).
    # Our optimal strategy is to select 1, 2, ..., floor(k // 2), and then
    # choose k, k + 1, ... if necessary, as selecting any number in the range
    # [ceil(k // 2), k - 1] will result in a pair summing up to k.

    def trapezoid(a: int, b: int) -> int:
      """Returns sum(a..b)."""
      return (a + b) * (b - a + 1) // 2

    mid = k // 2  # floor(k // 2)
    if n <= mid:
      return trapezoid(1, n)
    return trapezoid(1, mid) + trapezoid(k, k + (n - mid - 1))


# Link: https://leetcode.com/problems/sort-transformed-array/description/
class Solution:
  def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
    n = len(nums)
    upward = a > 0
    ans = [0] * n

    # The concavity of f only depends on a's sign.
    def f(x: int, a: int, b: int, c: int) -> int:
      return (a * x + b) * x + c

    quad = [f(num, a, b, c) for num in nums]

    i = n - 1 if upward else 0
    l = 0
    r = n - 1
    while l <= r:
      if upward:  # is the maximum in the both ends
        if quad[l] > quad[r]:
          ans[i] = quad[l]
          l += 1
        else:
          ans[i] = quad[r]
          r -= 1
        i -= 1
      else:  # is the minimum in the both ends
        if quad[l] < quad[r]:
          ans[i] = quad[l]
          l += 1
        else:
          ans[i] = quad[r]
          r -= 1
        i += 1

    return ans


# Link: https://leetcode.com/problems/matrix-block-sum/description/
class Solution:
  def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
    m = len(mat)
    n = len(mat[0])
    ans = [[0] * n for _ in range(m)]
    prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        prefix[i + 1][j + 1] = mat[i][j] + \
            prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j]

    for i in range(m):
      for j in range(n):
        r1 = max(0, i - k) + 1
        c1 = max(0, j - k) + 1
        r2 = min(m - 1, i + k) + 1
        c2 = min(n - 1, j + k) + 1
        ans[i][j] = prefix[r2][c2] - prefix[r2][c1 - 1] - \
            prefix[r1 - 1][c2] + prefix[r1 - 1][c1 - 1]

    return ans


# Link: https://leetcode.com/problems/minimum-cost-of-a-path-with-special-roads/description/
class Solution:
  def minimumCost(self, start: List[int], target: List[int], specialRoads: List[List[int]]) -> int:
    return self.dijkstra(specialRoads, *start, *target)

  def dijkstra(self, specialRoads: List[List[int]], srcX: int, srcY: int, dstX: int, dstY: int) -> int:
    n = len(specialRoads)
    # dist[i] := the minimum distance of (srcX, srcY) to specialRoads[i](x2, y2)
    dist = [math.inf] * n
    minHeap = []  # (d, u),(d, u), where u := the i-th specialRoads

    # (srcX, srcY) -> (x1, y1) to cost -> (x2, y2)
    for u, (x1, y1, _, _, cost) in enumerate(specialRoads):
      d = abs(x1 - srcX) + abs(y1 - srcY) + cost
      dist[u] = d
      heapq.heappush(minHeap, (dist[u], u))

    while minHeap:
      d, u = heapq.heappop(minHeap)
      _, _, ux2, uy2, _ = specialRoads[u]
      for v in range(n):
        if v == u:
          continue
        vx1, vy1, _, _, vcost = specialRoads[v]
        # (ux2, uy2) -> (vx1, vy1) to vcost -> (vx2, vy2)
        newDist = d + abs(vx1 - ux2) + abs(vy1 - uy2) + vcost
        if newDist < dist[v]:
          dist[v] = newDist
          heapq.heappush(minHeap, (dist[v], v))

    ans = abs(dstX - srcX) + abs(dstY - srcY)
    for u in range(n):
      _, _, x2, y2, _ = specialRoads[u]
      # (srcX, srcY) -> (x2, y2) -> (dstX, dstY).
      ans = min(ans, dist[u] + abs(dstX - x2) + abs(dstY - y2))

    return ans


# Link: https://leetcode.com/problems/partition-list/description/
class Solution:
  def partition(self, head: ListNode, x: int) -> ListNode:
    beforeHead = ListNode(0)
    afterHead = ListNode(0)
    before = beforeHead
    after = afterHead

    while head:
      if head.val < x:
        before.next = head
        before = head
      else:
        after.next = head
        after = head
      head = head.next

    after.next = None
    before.next = afterHead.next

    return beforeHead.next


# Link: https://leetcode.com/problems/vowel-spellchecker/description/
class Solution:
  def spellchecker(self, wordlist: List[str], queries: List[str]) -> List[str]:
    def lowerKey(word: str) -> str:
      return '$' + ''.join([c.lower() for c in word])

    def vowelKey(word: str) -> str:
      return ''.join(['*' if c.lower() in 'aeiou' else c.lower() for c in word])

    ans = []
    dict = {}

    for word in wordlist:
      dict.setdefault(word, word)
      dict.setdefault(lowerKey(word), word)
      dict.setdefault(vowelKey(word), word)

    for query in queries:
      if query in dict:
        ans.append(dict[query])
      elif lowerKey(query) in dict:
        ans.append(dict[lowerKey(query)])
      elif vowelKey(query) in dict:
        ans.append(dict[vowelKey(query)])
      else:
        ans.append('')

    return ans


# Link: https://leetcode.com/problems/four-divisors/description/
class Solution:
  def sumFourDivisors(self, nums: List[int]) -> int:
    ans = 0

    for num in nums:
      divisor = 0
      for i in range(2, int(sqrt(num)) + 1):
        if num % i == 0:
          if divisor == 0:
            divisor = i
          else:
            divisor = 0
            break
      if divisor > 0 and divisor * divisor < num:
        ans += 1 + num + divisor + num // divisor

    return ans


# Link: https://leetcode.com/problems/boats-to-save-people/description/
class Solution:
  def numRescueBoats(self, people: List[int], limit: int) -> int:
    ans = 0
    i = 0
    j = len(people) - 1

    people.sort()

    while i <= j:
      remain = limit - people[j]
      j -= 1
      if people[i] <= remain:
        i += 1
      ans += 1

    return ans


# Link: https://leetcode.com/problems/sum-of-two-integers/description/
class Solution:
  def getSum(self, a: int, b: int) -> int:
    mask = 0xFFFFFFFF
    kMax = 2000

    while b != 0:
      a, b = (a ^ b) & mask, ((a & b) << 1) & mask

    return a if a < kMax else ~(a ^ mask)


# Link: https://leetcode.com/problems/array-of-doubled-pairs/description/
class Solution:
  def canReorderDoubled(self, arr: List[int]) -> bool:
    count = collections.Counter(arr)

    for key in sorted(count, key=abs):
      if count[key] > count[2 * key]:
        return False
      count[2 * key] -= count[key]

    return True


# Link: https://leetcode.com/problems/range-addition/description/
class Solution:
  def getModifiedArray(self, length: int, updates: list[list[int]]) -> list[int]:
    line = [0] * length

    for start, end, inc in updates:
      line[start] += inc
      if end + 1 < length:
        line[end + 1] -= inc

    return itertools.accumulate(line)


# Link: https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/
class Solution:
  def getMoneyAmount(self, n: int) -> int:
    @functools.lru_cache(None)
    def dp(i: int, j: int) -> int:
      """Returns the minimum money you need to guarantee a win of picking i..j.
      """
      if i >= j:
        return 0
      return min(max(dp(i, k - 1), dp(k + 1, j)) + k
                 for k in range(i, j + 1))

    return dp(1, n)


# Link: https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/
class Solution:
  def getMoneyAmount(self, n: int) -> int:
    # dp[i][j] := the minimum money you need to guarantee a win of picking i..j
    dp = [[0] * (n + 2) for _ in range(n + 2)]

    for d in range(1, n + 1):
      for i in range(1, n - d + 1):
        j = i + d
        dp[i][j] = math.inf
        for k in range(i, j + 1):
          dp[i][j] = min(dp[i][j], max(dp[i][k - 1], dp[k + 1][j]) + k)

    return dp[1][n]


# Link: https://leetcode.com/problems/multiply-strings/description/
class Solution:
  def multiply(self, num1: str, num2: str) -> str:
    s = [0] * (len(num1) + len(num2))

    for i in reversed(range(len(num1))):
      for j in reversed(range(len(num2))):
        mult = int(num1[i]) * int(num2[j])
        summ = mult + s[i + j + 1]
        s[i + j] += summ // 10
        s[i + j + 1] = summ % 10

    for i, c in enumerate(s):
      if c != 0:
        break

    return ''.join(map(str, s[i:]))


# Link: https://leetcode.com/problems/minimizing-array-after-replacing-pairs-with-their-product/description/
class Solution:
  def minArrayLength(self, nums: List[int], k: int) -> int:
    count = 0
    prod = -1

    for num in nums:
      if num == 0:
        return 1
      if prod != -1 and prod * num <= k:
        prod *= num
      else:
        prod = num
        count += 1

    return count


# Link: https://leetcode.com/problems/find-triangular-sum-of-an-array/description/
class Solution:
  def triangularSum(self, nums: List[int]) -> int:
    for sz in range(len(nums), 0, -1):
      for i in range(sz - 1):
        nums[i] = (nums[i] + nums[i + 1]) % 10
    return nums[0]


# Link: https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/description/
class Solution:
  def removeDuplicates(self, s: str, k: int) -> str:
    stack = []

    for c in s:
      if not stack or stack[-1][0] != c:
        stack.append([c, 1])
      else:  # stack[-1][0] == c
        stack[-1][1] += 1
        if stack[-1][1] == k:
          stack.pop()

    return ''.join(c * count for c, count in stack)


# Link: https://leetcode.com/problems/detect-squares/description/
class DetectSquares:
  def __init__(self):
    self.pointCount = collections.Counter()

  def add(self, point: List[int]) -> None:
    self.pointCount[tuple(point)] += 1

  def count(self, point: List[int]) -> int:
    x1, y1 = point
    ans = 0
    for (x3, y3), c in self.pointCount.items():
      if x1 != x3 and abs(x1 - x3) == abs(y1 - y3):
        ans += c * self.pointCount[(x1, y3)] * self.pointCount[(x3, y1)]
    return ans


# Link: https://leetcode.com/problems/random-point-in-non-overlapping-rectangles/description/
class Solution:
  def __init__(self, rects: List[List[int]]):
    self.rects = rects
    self.areas = list(itertools.accumulate(
        [(x2 - x1 + 1) * (y2 - y1 + 1) for x1, y1, x2, y2 in rects]))

  def pick(self) -> List[int]:
    index = bisect_right(self.areas, random.randint(0, self.areas[-1] - 1))
    x1, y1, x2, y2 = self.rects[index]
    return [random.randint(x1, x2), random.randint(y1, y2)]


# Link: https://leetcode.com/problems/count-the-number-of-beautiful-subarrays/description/
class Solution:
  def beautifulSubarrays(self, nums: List[int]) -> int:
    # A subarray is beautiful if xor(subarray) = 0.
    ans = 0
    prefix = 0
    prefixCount = collections.Counter({0: 1})

    for num in nums:
      prefix ^= num
      ans += prefixCount[prefix]
      prefixCount[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/description/
class Solution:
  def countMaxOrSubsets(self, nums: List[int]) -> int:
    ors = functools.reduce(operator.or_, nums)
    ans = 0

    def dfs(i: int, path: int) -> None:
      nonlocal ans
      if i == len(nums):
        if path == ors:
          ans += 1
        return

      dfs(i + 1, path)
      dfs(i + 1, path | nums[i])

    dfs(0, 0)
    return ans


# Link: https://leetcode.com/problems/count-number-of-ways-to-place-houses/description/
class Solution:
  def countHousePlacements(self, n: int) -> int:
    kMod = 1_000_000_007
    house = 1  # the number of ways ending in a house
    space = 1  # the number of ways ending in a space
    total = house + space

    for _ in range(2, n + 1):
      house = space
      space = total
      total = (house + space) % kMod

    return total**2 % kMod


# Link: https://leetcode.com/problems/design-front-middle-back-queue/description/
class FrontMiddleBackQueue:
  def __init__(self):
    self.frontQueue = collections.deque()
    self.backQueue = collections.deque()

  def pushFront(self, val: int) -> None:
    self.frontQueue.appendleft(val)
    self._moveFrontToBackIfNeeded()

  def pushMiddle(self, val: int) -> None:
    if len(self.frontQueue) == len(self.backQueue):
      self.backQueue.appendleft(val)
    else:
      self.frontQueue.append(val)

  def pushBack(self, val: int) -> None:
    self.backQueue.append(val)
    self._moveBackToFrontIfNeeded()

  def popFront(self) -> int:
    if self.frontQueue:
      x = self.frontQueue.popleft()
      self._moveBackToFrontIfNeeded()
      return x
    if self.backQueue:
      return self.backQueue.popleft()
    return -1

  def popMiddle(self) -> int:
    if not self.frontQueue and not self.backQueue:
      return -1
    if len(self.frontQueue) + 1 == len(self.backQueue):
      return self.backQueue.popleft()
    return self.frontQueue.pop()

  def popBack(self) -> int:
    if self.backQueue:
      x = self.backQueue.pop()
      self._moveFrontToBackIfNeeded()
      return x
    return -1

  def _moveFrontToBackIfNeeded(self) -> None:
    if len(self.frontQueue) - 1 == len(self.backQueue):
      self.backQueue.appendleft(self.frontQueue.pop())

  def _moveBackToFrontIfNeeded(self) -> None:
    if len(self.frontQueue) + 2 == len(self.backQueue):
      self.frontQueue.append(self.backQueue.popleft())


# Link: https://leetcode.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/description/
class Solution:
  def findMinFibonacciNumbers(self, k: int) -> int:
    if k < 2:  # k == 0 or k == 1
      return k

    a = 1  # F_1
    b = 1  # F_2

    while b <= k:
      #    a, b = F_{i + 1}, F_{i + 2}
      # -> a, b = F_{i + 2}, F_{i + 3}
      a, b = b, a + b

    return 1 + self.findMinFibonacciNumbers(k - a)


# Link: https://leetcode.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/description/
class Solution:
  def findMinFibonacciNumbers(self, k: int) -> int:
    ans = 0
    a = 1  # F_1
    b = 1  # F_2

    while b <= k:
      #    a, b = F_{i + 1}, F_{i + 2}
      # -> a, b = F_{i + 2}, F_{i + 3}
      a, b = b, a + b

    while a > 0:
      if a <= k:
        k -= a
        ans += 1
      #    a, b = F_{i + 2}, F_{i + 3}
      # -> a, b = F_{i + 1}, F_{i + 2}
      a, b = b - a, a

    return ans


# Link: https://leetcode.com/problems/design-file-system/description/
class TrieNode:
  def __init__(self, value: int = 0):
    self.children: Dict[str, TrieNode] = {}
    self.value = value


class FileSystem:
  def __init__(self):
    self.root = TrieNode()

  def createPath(self, path: str, value: int) -> bool:
    node: TrieNode = self.root
    subpaths = path.split('/')

    for i in range(1, len(subpaths) - 1):
      if subpaths[i] not in node.children:
        return False
      node = node.children[subpaths[i]]

    if subpaths[-1] in node.children:
      return False
    node.children[subpaths[-1]] = TrieNode(value)
    return True

  def get(self, path: str) -> int:
    node: TrieNode = self.root

    for subpath in path.split('/')[1:]:
      if subpath not in node.children:
        return -1
      node = node.children[subpath]

    return node.value


# Link: https://leetcode.com/problems/restore-ip-addresses/description/
class Solution:
  def restoreIpAddresses(self, s: str) -> List[str]:
    ans = []

    def dfs(start: int, path: List[int]) -> None:
      if len(path) == 4 and start == len(s):
        ans.append(path[0] + '.' + path[1] + '.' + path[2] + '.' + path[3])
        return
      if len(path) == 4 or start == len(s):
        return

      for length in range(1, 4):
        if start + length > len(s):
          return  # out-of-bounds
        if length > 1 and s[start] == '0':
          return  # leading '0'
        num = s[start: start + length]
        if int(num) > 255:
          return
        dfs(start + length, path + [num])

    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/strings-differ-by-one-character/description/
class Solution:
  def differByOne(self, dict: List[str]) -> bool:
    kMod = 1_000_000_007
    m = len(dict[0])

    def val(c: str) -> int:
      return ord(c) - ord('a')

    def getHash(s: str) -> int:
      """Returns the hash of `s`. Assume the length of `s` is m.

      e.g. getHash(s) = 26^(m - 1) * s[0] + 26^(m - 2) * s[1] + ... + s[m - 1].
      """
      hash = 0
      for c in s:
        hash = (26 * hash + val(c))
      return hash

    wordToHash = [getHash(word) for word in dict]

    # Compute the hash without each letter.
    # e.g. hash of "abc" = 26^2 * 'a' + 26 * 'b' + 'c'
    #   newHash of "a*c" = hash - 26 * 'b'
    coefficient = 1
    for j in range(m - 1, -1, -1):
      newHashToIndices = collections.defaultdict(list)
      for i, (word, hash) in enumerate(zip(dict, wordToHash)):
        newHash = (hash - coefficient * val(word[j]) % kMod + kMod) % kMod
        if any(word[:j] == dict[index][:j] and word[j + 1:] == dict[index][j + 1:]
               for index in newHashToIndices[newHash]):
          return True
        newHashToIndices[newHash].append(i)
      coefficient = (26 * coefficient) % kMod

    return False


# Link: https://leetcode.com/problems/adding-two-negabinary-numbers/description/
class Solution:
  def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
    ans = []
    carry = 0

    while carry or arr1 or arr2:
      if arr1:
        carry += arr1.pop()
      if arr2:
        carry += arr2.pop()
      ans.append(carry & 1)
      carry = -(carry >> 1)

    while len(ans) > 1 and ans[-1] == 0:
      ans.pop()

    return ans[::-1]


# Link: https://leetcode.com/problems/decode-string/description/
class Solution:
  def decodeString(self, s: str) -> str:
    ans = ''

    while self.i < len(s) and s[self.i] != ']':
      if s[self.i].isdigit():
        k = 0
        while self.i < len(s) and s[self.i].isdigit():
          k = k * 10 + int(s[self.i])
          self.i += 1
        self.i += 1  # '['
        decodedString = self.decodeString(s)
        self.i += 1  # ']'
        ans += k * decodedString
      else:
        ans += s[self.i]
        self.i += 1

    return ans

  i = 0


# Link: https://leetcode.com/problems/decode-string/description/
class Solution:
  def decodeString(self, s: str) -> str:
    stack = []  # (prevStr, repeatCount)
    currStr = ''
    currNum = 0

    for c in s:
      if c.isdigit():
        currNum = currNum * 10 + int(c)
      else:
        if c == '[':
          stack.append((currStr, currNum))
          currStr = ''
          currNum = 0
        elif c == ']':
          prevStr, num = stack.pop()
          currStr = prevStr + num * currStr
        else:
          currStr += c

    return currStr


# Link: https://leetcode.com/problems/online-stock-span/description/
class StockSpanner:
  def __init__(self):
    self.stack = []  # (price, span)

  def next(self, price: int) -> int:
    span = 1
    while self.stack and self.stack[-1][0] <= price:
      span += self.stack.pop()[1]
    self.stack.append((price, span))
    return span


# Link: https://leetcode.com/problems/find-all-groups-of-farmland/description/
class Solution:
  def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
    ans = []

    def dfs(i: int, j: int, cell: List[int]) -> None:
      if i < 0 or i == len(land) or j < 0 or j == len(land[0]):
        return
      if land[i][j] != 1:
        return
      land[i][j] = 2  # Mark as visited.
      cell[0] = max(cell[0], i)
      cell[1] = max(cell[1], j)
      dfs(i + 1, j, cell)
      dfs(i, j + 1, cell)

    for i in range(len(land)):
      for j in range(len(land[0])):
        if land[i][j] == 1:
          cell = [i, j]
          dfs(i, j, cell)
          ans.append([i, j, *cell])

    return ans


# Link: https://leetcode.com/problems/combinations/description/
class Solution:
  def combine(self, n: int, k: int) -> List[List[int]]:
    ans = []

    def dfs(s: int, path: List[int]) -> None:
      if len(path) == k:
        ans.append(path.copy())
        return

      for i in range(s, n + 1):
        path.append(i)
        dfs(i + 1, path)
        path.pop()

    dfs(1, [])
    return ans


# Link: https://leetcode.com/problems/tweet-counts-per-frequency/description/
from sortedcontainers import SortedList


class TweetCounts:
  def __init__(self):
    self.tweetNameToTimes = collections.defaultdict(SortedList)

  def recordTweet(self, tweetName: str, time: int) -> None:
    self.tweetNameToTimes[tweetName].add(time)

  def getTweetCountsPerFrequency(self, freq: str, tweetName: str, startTime: int, endTime: int) -> List[int]:
    counts = []
    times = self.tweetNameToTimes[tweetName]
    chunk = 60 if freq == 'minute' else 3600 if freq == 'hour' else 86400

    # I := startTime of each chunk
    for i in range(startTime, endTime + 1, chunk):
      j = min(i + chunk, endTime + 1)  # EndTime of each chunk
      counts.append(bisect_left(times, j) - bisect_left(times, i))

    return counts


# Link: https://leetcode.com/problems/airplane-seat-assignment-probability/description/
class Solution:
  def nthPersonGetsNthSeat(self, n: int) -> float:
    return 1 if n == 1 else 0.5


# Link: https://leetcode.com/problems/count-pairs-of-points-with-distance-k/description/
class Solution:
  def countPairs(self, coordinates: List[List[int]], k: int) -> int:
    ans = 0

    for x in range(k + 1):
      y = k - x
      count = collections.Counter()
      for xi, yi in coordinates:
        ans += count[(xi ^ x, yi ^ y)]
        count[(xi, yi)] += 1

    return ans


# Link: https://leetcode.com/problems/longest-common-subsequence-between-sorted-arrays/description/
class Solution:
  def longestCommonSubsequence(self, arrays: List[List[int]]) -> List[int]:
    kMax = 100
    ans = []
    count = [0] * (kMax + 1)

    for array in arrays:
      for a in array:
        count[a] += 1
        if count[a] == len(arrays):
          ans.append(a)

    return ans


# Link: https://leetcode.com/problems/longest-common-subsequence-between-sorted-arrays/description/
class Solution:
  def longestCommonSubsequence(self, arrays: List[List[int]]) -> List[int]:
    return sorted(functools.reduce(lambda a, b: set(a) & set(b), arrays))


# Link: https://leetcode.com/problems/binary-tree-upside-down/description/
class Solution:
  def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    prevRoot = None
    prevRightChild = None

    while root:
      nextRoot = root.left  # Cache the next root.
      root.left = prevRightChild
      prevRightChild = root.right
      root.right = prevRoot
      prevRoot = root  # Record the previous root.
      root = nextRoot  # Update the root.

    return prevRoot


# Link: https://leetcode.com/problems/binary-tree-upside-down/description/
class Solution:
  def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root or not root.left:
      return root

    newRoot = self.upsideDownBinaryTree(root.left)
    root.left.left = root.right  # 2's left = 3 (root's right)
    root.left.right = root  # 2's right = 1 (root)
    root.left = None
    root.right = None
    return newRoot


# Link: https://leetcode.com/problems/combination-sum-iii/description/
class Solution:
  def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    ans = []

    def dfs(k: int, n: int, s: int, path: List[int]) -> None:
      if k == 0 and n == 0:
        ans.append(path)
        return
      if k == 0 or n < 0:
        return

      for i in range(s, 10):
        dfs(k - 1, n - i, i + 1, path + [i])

    dfs(k, n, 1, [])
    return ans


# Link: https://leetcode.com/problems/binary-trees-with-factors/description/
class Solution:
  def numFactoredBinaryTrees(self, arr: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(arr)
    # dp[i] := the number of binary trees with arr[i] as the root
    dp = [1] * n
    arr.sort()
    numToIndex = {a: i for i, a in enumerate(arr)}

    for i, root in enumerate(arr):  # arr[i] is the root
      for j in range(i):
        if root % arr[j] == 0:  # arr[j] is the left subtree
          right = root // arr[j]
          if right in numToIndex:
            dp[i] += dp[j] * dp[numToIndex[right]]
            dp[i] %= kMod

    return sum(dp) % kMod


# Link: https://leetcode.com/problems/add-minimum-number-of-rungs/description/
class Solution:
  def addRungs(self, rungs: List[int], dist: int) -> int:
    ans = 0
    prev = 0

    for rung in rungs:
      ans += (rung - prev - 1) // dist
      prev = rung

    return ans


# Link: https://leetcode.com/problems/find-closest-node-to-given-two-nodes/description/
class Solution:
  def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
    kMax = 10000
    dist1 = self._getDist(edges, node1)
    dist2 = self._getDist(edges, node2)
    minDist = kMax
    ans = -1

    for i, (d1, d2) in enumerate(zip(dist1, dist2)):
      if min(d1, d2) >= 0:
        maxDist = max(d1, d2)
        if maxDist < minDist:
          minDist = maxDist
          ans = i

    return ans

  def _getDist(self, edges: List[int], u: int) -> List[int]:
    dist = [-1] * len(edges)
    d = 0
    while u != -1 and dist[u] == -1:
      dist[u] = d
      d += 1
      u = edges[u]
    return dist


# Link: https://leetcode.com/problems/coordinate-with-maximum-network-quality/description/
class Solution:
  def bestCoordinate(self, towers: List[List[int]], radius: int) -> List[int]:
    kMax = 50
    n = len(towers)
    ans = [0] * 2
    maxQuality = 0

    def dist(tower: List[int], i: int, j: int) -> float:
      """Returns the distance between the tower and the coordinate."""
      return math.sqrt((tower[0] - i)**2 + (tower[1] - j)**2)

    for i in range(kMax + 1):
      for j in range(kMax + 1):
        qualitySum = 0
        for tower in towers:
          q = tower[2]
          d = dist(tower, i, j)
          if d <= radius:
            qualitySum += int(q / (1 + d))
        if qualitySum > maxQuality:
          maxQuality = qualitySum
          ans = [i, j]

    return ans


# Link: https://leetcode.com/problems/shortest-word-distance-iii/description/
class Solution:
  def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
    isSame = word1 == word2
    ans = math.inf
    # If word1 == word2, index1 is the newest index.
    index1 = len(wordsDict)
    # If word1 == word2, index2 is the previous index.
    index2 = -len(wordsDict)

    for i, word in enumerate(wordsDict):
      if word == word1:
        if isSame:
          index2 = index1
        index1 = i
      elif word == word2:
        index2 = i
      ans = min(ans, abs(index1 - index2))

    return ans


# Link: https://leetcode.com/problems/maximum-linear-stock-score/description/
class Solution:
  def maxScore(self, prices: List[int]) -> int:
    groupIdToSum = collections.defaultdict(int)

    for i, price in enumerate(prices):
      groupIdToSum[price - i] += price

    return max(groupIdToSum.values())


# Link: https://leetcode.com/problems/minimum-numbers-of-function-calls-to-make-target-array/description/
class Solution:
  def minOperations(self, nums: List[int]) -> int:
    return sum(bin(num).count('1') for num in nums) + len(bin(max(nums))) - 3


# Link: https://leetcode.com/problems/super-ugly-number/description/
class Solution:
  def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
    k = len(primes)
    nums = [1]
    indices = [0] * k

    while len(nums) < n:
      nexts = [0] * k
      for i in range(k):
        nexts[i] = nums[indices[i]] * primes[i]
      next = min(nexts)
      for i in range(k):
        if next == nexts[i]:
          indices[i] += 1
      nums.append(next)

    return nums[-1]


# Link: https://leetcode.com/problems/super-ugly-number/description/
class UglyNum:
  def __init__(self, prime: int, index: int, value: int):
    self.prime = prime
    self.index = index  # Point the next index of uglyNums.
    self.value = value  # prime * uglyNums[index]


class Solution:
  def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
    minHeap = []  # (value, prime, index)
    uglyNums = [1]

    for prime in primes:
      heapq.heappush(minHeap, (prime * uglyNums[0], prime, 1))

    while len(uglyNums) < n:
      uglyNums.append(minHeap[0][0])
      while minHeap[0][0] == uglyNums[-1]:
        _, prime, index = heapq.heappop(minHeap)
        heapq.heappush(minHeap, (prime * uglyNums[index], prime, index + 1))

    return uglyNums[-1]


# Link: https://leetcode.com/problems/choose-edges-to-maximize-score-in-a-tree/description/
class Solution:
  def maxScore(self, edges: List[List[int]]) -> int:
    n = len(edges)
    graph = [[] for _ in range(n)]

    for i, (parent, weight) in enumerate(edges):
      if parent != -1:
        graph[parent].append((i, weight))

    takeRoot, notTakeRoot = self._dfs(graph, 0)
    return max(takeRoot, notTakeRoot)

  def _dfs(self, graph: List[List[int]], u: int) -> Tuple[int, int]:
    """
    Returns (the maximum sum at u if we take one u->v edge,
             the maximum sum at u if we don't take any child edge).
    """
    bestEdge = 0
    notTakeU = 0

    for v, w in graph[u]:
      takeV, notTakeV = self._dfs(graph, v)
      bestEdge = max(bestEdge, w + notTakeV - takeV)
      notTakeU += takeV

    return (bestEdge + notTakeU, notTakeU)


# Link: https://leetcode.com/problems/smallest-greater-multiple-made-of-two-digits/description/
class Solution:
  def findInteger(self, k: int, digit1: int, digit2: int) -> int:
    minDigit = min(digit1, digit2)
    maxDigit = max(digit1, digit2)
    digits = [minDigit] if minDigit == maxDigit else [minDigit, maxDigit]
    q = collections.deque()

    for digit in digits:
      q.append(digit)

    while q:
      u = q.popleft()
      if u > k and u % k == 0:
        return u
      if u == 0:
        continue
      for digit in digits:
        nextNum = u * 10 + digit
        if nextNum > 2**31 - 1:
          continue
        q.append(nextNum)

    return -1


# Link: https://leetcode.com/problems/smallest-greater-multiple-made-of-two-digits/description/
class Solution:
  def findInteger(self, k: int, digit1: int, digit2: int) -> int:
    def dfs(x: int) -> int:
      if x > 2**31 - 1:
        return -1
      if x > k and x % k == 0:
        return x
      # Skip if digit1/digit2 and x are zero.
      a = -1 if x + digit1 == 0 else dfs(x * 10 + digit1)
      b = -1 if x + digit2 == 0 else dfs(x * 10 + digit2)
      if a == -1:
        return b
      if b == -1:
        return a
      return min(a, b)

    return dfs(0)


# Link: https://leetcode.com/problems/magic-squares-in-grid/description/
class Solution:
  def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
    def isMagic(i: int, j: int) -> int:
      s = "".join(str(grid[i + num // 3][j + num % 3])
                  for num in [0, 1, 2, 5, 8, 7, 6, 3])
      return s in "43816729" * 2 or s in "43816729"[::-1] * 2

    ans = 0

    for i in range(len(grid) - 2):
      for j in range(len(grid[0]) - 2):
        if grid[i][j] % 2 == 0 and grid[i + 1][j + 1] == 5:
          ans += isMagic(i, j)

    return ans


# Link: https://leetcode.com/problems/check-if-a-string-is-a-valid-sequence-from-root-to-leaves-path-in-a-binary-tree/description/
class Solution:
  def isValidSequence(self, root: Optional[TreeNode], arr: List[int]) -> bool:
    def isValidSequence(root: Optional[TreeNode], i: int) -> bool:
      if not root:
        return False
      if i == len(arr) - 1:
        return root.val == arr[i] and not root.left and not root.right
      return root.val == arr[i] and (isValidSequence(root.left, i + 1) or isValidSequence(root.right,  i + 1))

    return isValidSequence(root, 0)


# Link: https://leetcode.com/problems/all-possible-full-binary-trees/description/
class Solution:
  @functools.lru_cache(None)
  def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    if n % 2 == 0:
      return []
    if n == 1:
      return [TreeNode(0)]

    ans = []

    for leftCount in range(n):
      rightCount = n - 1 - leftCount
      for left in self.allPossibleFBT(leftCount):
        for right in self.allPossibleFBT(rightCount):
          ans.append(TreeNode(0))
          ans[-1].left = left
          ans[-1].right = right

    return ans


# Link: https://leetcode.com/problems/4sum/description/
class Solution:
  def fourSum(self, nums: List[int], target: int):
    ans = []

    def nSum(l: int, r: int, target: int, n: int, path: List[int], ans: List[List[int]]) -> None:
      """Finds n numbers that add up to the target in [l, r]."""
      if r - l + 1 < n or n < 2 or target < nums[l] * n or target > nums[r] * n:
        return
      if n == 2:
        while l < r:
          summ = nums[l] + nums[r]
          if summ == target:
            ans.append(path + [nums[l], nums[r]])
            l += 1
            while nums[l] == nums[l - 1] and l < r:
              l += 1
          elif summ < target:
            l += 1
          else:
            r -= 1
        return

      for i in range(l, r + 1):
        if i > l and nums[i] == nums[i - 1]:
          continue

        nSum(i + 1, r, target - nums[i], n - 1, path + [nums[i]], ans)

    nums.sort()
    nSum(0, len(nums) - 1, target, 4, [], ans)
    return ans


# Link: https://leetcode.com/problems/bomb-enemy/description/
class Solution:
  def maxKilledEnemies(self, grid: List[List[chr]]) -> int:
    m = len(grid)
    n = len(grid[0])
    enemyCount = 0
    # dp[i][j] := the maximum enemies grid[i][j] can kill
    dp = [[0] * n for _ in range(m)]

    def update(i: int, j: int) -> None:
      nonlocal enemyCount
      if grid[i][j] == '0':
        dp[i][j] += enemyCount
      elif grid[i][j] == 'E':
        enemyCount += 1
      else:  # grid[i][j] == 'W'
        enemyCount = 0

    # Extend the four directions, if meet 'W', need to start over from 0.
    for i in range(m):
      enemyCount = 0
      for j in range(n):
        update(i, j)
      enemyCount = 0
      for j in reversed(range(n)):
        update(i, j)

    for j in range(n):
      enemyCount = 0
      for i in range(m):
        update(i, j)
      enemyCount = 0
      for i in reversed(range(m)):
        update(i, j)

    # Returns sum(map(sum, dp))
    return max(map(max, dp))


# Link: https://leetcode.com/problems/spiral-matrix-ii/description/
class Solution:
  def generateMatrix(self, n: int) -> List[List[int]]:
    ans = [[0] * n for _ in range(n)]
    count = 1

    for min in range(n // 2):
      max = n - min - 1
      for i in range(min, max):
        ans[min][i] = count
        count += 1
      for i in range(min, max):
        ans[i][max] = count
        count += 1
      for i in range(max, min, -1):
        ans[max][i] = count
        count += 1
      for i in range(max, min, -1):
        ans[i][min] = count
        count += 1

    if n & 1:
      ans[n // 2][n // 2] = count

    return ans


# Link: https://leetcode.com/problems/find-the-safest-path-in-a-grid/description/
class Solution:
  def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
    self.dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    n = len(grid)
    distToThief = self._getDistToThief(grid)

    def hasValidPath(safeness: int) -> bool:
      if distToThief[0][0] < safeness:
        return False

      q = collections.deque([(0, 0)])
      seen = {(0, 0)}

      while q:
        i, j = q.popleft()
        if distToThief[i][j] < safeness:
          continue
        if i == n - 1 and j == n - 1:
          return True
        for dx, dy in self.dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == n or y < 0 or y == n:
            continue
          if (x, y) in seen:
            continue
          q.append((x, y))
          seen.add((x, y))

      return False

    return bisect.bisect_left(range(n * 2), True,
                              key=lambda m: not hasValidPath(m)) - 1

  def _getDistToThief(self, grid: List[List[int]]) -> List[List[int]]:
    n = len(grid)
    distToThief = [[0] * n for _ in range(n)]
    q = collections.deque()
    seen = set()

    for i in range(n):
      for j in range(n):
        if grid[i][j] == 1:
          q.append((i, j))
          seen.add((i, j))

    dist = 0
    while q:
      for _ in range(len(q)):
        i, j = q.popleft()
        distToThief[i][j] = dist
        for dx, dy in self.dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == n or y < 0 or y == n:
            continue
          if (x, y) in seen:
            continue
          q.append((x, y))
          seen.add((x, y))
      dist += 1

    return distToThief


# Link: https://leetcode.com/problems/sum-of-numbers-with-units-digit-k/description/
class Solution:
  def minimumNumbers(self, num: int, k: int) -> int:
    if num == 0:
      return 0

    # Assume the size of the set is n, and the numbers in the set are X1, X2,
    # ..., Xn. Since the units digit of each number is k, X1 + X2 + ... + Xn =
    # N * k + 10 * (x1 + x2 + ... + xn) = num. Therefore, the goal is to find
    # the n s.t. n * k % 10 = num % 10
    for i in range(1, 11):
      if i * k > num + 1:
        break
      if i * k % 10 == num % 10:
        return i

    return -1


# Link: https://leetcode.com/problems/triangle/description/
class Solution:
  def minimumTotal(self, triangle: List[List[int]]) -> int:
    for i in reversed(range(len(triangle) - 1)):
      for j in range(i + 1):
        triangle[i][j] += min(triangle[i + 1][j],
                              triangle[i + 1][j + 1])

    return triangle[0][0]


# Link: https://leetcode.com/problems/simplified-fractions/description/
class Solution:
  def simplifiedFractions(self, n: int) -> List[str]:
    ans = []
    for denominator in range(2, n + 1):
      for numerator in range(1, denominator):
        if math.gcd(denominator, numerator) == 1:
          ans.append(str(numerator) + '/' + str(denominator))
    return ans


# Link: https://leetcode.com/problems/move-pieces-to-obtain-a-string/description/
class Solution:
  def canChange(self, start: str, target: str) -> bool:
    n = len(start)
    i = 0  # start's index
    j = 0  # target's index

    while i <= n and j <= n:
      while i < n and start[i] == '_':
        i += 1
      while j < n and target[j] == '_':
        j += 1
      if i == n or j == n:
        return i == n and j == n
      if start[i] != target[j]:
        return False
      if start[i] == 'R' and i > j:
        return False
      if start[i] == 'L' and i < j:
        return False
      i += 1
      j += 1

    return True


# Link: https://leetcode.com/problems/linked-list-in-binary-tree/description/
class Solution:
  def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
    if not root:
      return False
    return self._isContinuousSubPath(head, root) or \
        self.isSubPath(head, root.left) or \
        self.isSubPath(head, root.right)

  def _isContinuousSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
    if not head:
      return True
    if not root:
      return False
    return head.val == root.val and \
        (self._isContinuousSubPath(head.next, root.left)
         or self._isContinuousSubPath(head.next, root.right))


# Link: https://leetcode.com/problems/product-of-the-last-k-numbers/description/
class ProductOfNumbers:
  def __init__(self):
    self.prefix = [1]

  def add(self, num: int) -> None:
    if num == 0:
      self.prefix = [1]
    else:
      self.prefix.append(self.prefix[-1] * num)

  def getProduct(self, k: int) -> int:
    return 0 if k >= len(self.prefix) else self.prefix[-1] // self.prefix[len(self.prefix) - k - 1]


# Link: https://leetcode.com/problems/equal-tree-partition/description/
class Solution:
  def checkEqualTree(self, root: Optional[TreeNode]) -> bool:
    if not root:
      return False

    seen = set()

    def dfs(root: Optional[TreeNode]) -> int:
      if not root:
        return 0

      summ = root.val + dfs(root.left) + dfs(root.right)
      seen.add(summ)
      return summ

    summ = root.val + dfs(root.left) + dfs(root.right)
    return summ % 2 == 0 and summ // 2 in seen


# Link: https://leetcode.com/problems/path-with-maximum-probability/description/
class Solution:
  def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
    graph = [[] for _ in range(n)]  # {a: [(b, probability_ab)]}
    maxHeap = [(-1.0, start)]   # (the probability to reach u, u)
    seen = [False] * n

    for i, ((u, v), prob) in enumerate(zip(edges, succProb)):
      graph[u].append((v, prob))
      graph[v].append((u, prob))

    while maxHeap:
      prob, u = heapq.heappop(maxHeap)
      prob *= -1
      if u == end:
        return prob
      if seen[u]:
        continue
      seen[u] = True
      for nextNode, edgeProb in graph[u]:
        if seen[nextNode]:
          continue
        heapq.heappush(maxHeap, (-prob * edgeProb, nextNode))

    return 0


# Link: https://leetcode.com/problems/number-of-people-aware-of-a-secret/description/
class Solution:
  def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
    kMod = 1_000_000_007
    share = 0
    # dp[i] := the number of people know the secret at day i
    dp = [0] * n  # Maps day i to i + 1.
    dp[0] = 1

    for i in range(1, n):
      if i - delay >= 0:
        share += dp[i - delay]
      if i - forget >= 0:
        share -= dp[i - forget]
      share += kMod
      share %= kMod
      dp[i] = share

    # People before day `n - forget - 1` already forget the secret.
    return sum(dp[-forget:]) % kMod


# Link: https://leetcode.com/problems/find-the-punishment-number-of-an-integer/description/
class Solution:
  def punishmentNumber(self, n: int) -> int:
    def isPossible(accumulate: int, running: int, numChars: List[str], s: int, target: int) -> bool:
      """
      Returns True if the sum of any split of `numChars` equals to the target.
      """
      if s == len(numChars):
        return target == accumulate + running
      d = int(numChars[s])
      return (
          # Keep growing `running`.
          isPossible(accumulate, running * 10 + d, numChars, s + 1, target) or
          # Start a new `running`.
          isPossible(accumulate + running, d, numChars, s + 1, target)
      )

    return sum(i * i
               for i in range(1, n + 1)
               if isPossible(0, 0, str(i * i), 0, i))


# Link: https://leetcode.com/problems/flip-string-to-monotone-increasing/description/
class Solution:
  def minFlipsMonoIncr(self, s: str) -> int:
    # the number of characters to be flilpped to make the substring so far
    # monotone increasing
    dp = 0
    count1 = 0

    for c in s:
      if c == '0':
        # 1. Flip '0'.
        # 2. Keep '0' and flip all the previous 1s.
        dp = min(dp + 1, count1)
      else:
        count1 += 1

    return dp


# Link: https://leetcode.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/description/
class Solution:
  def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
    n = len(arr)
    l = 0
    r = n - 1

    # arr[0..l] is non-decreasing.
    while l < n - 1 and arr[l + 1] >= arr[l]:
      l += 1
    # arr[r..n - 1] is non-decreasing.
    while r > 0 and arr[r - 1] <= arr[r]:
      r -= 1
    # Remove arr[l + 1..n - 1] or arr[0..r - 1].
    ans = min(n - 1 - l, r)

    # Since arr[0..l] and arr[r..n - 1] are non-decreasing, we place pointers
    # at the rightmost indices, l and n - 1, and greedily shrink them toward
    # the leftmost indices, 0 and r, respectively. By removing arr[i + 1..j],
    # we ensure that `arr` becomes non-decreasing.
    i = l
    j = n - 1
    while i >= 0 and j >= r and j > i:
      if arr[i] <= arr[j]:
        j -= 1
      else:
        i -= 1
      ans = min(ans, j - i)

    return ans


# Link: https://leetcode.com/problems/maximum-or/description/
class Solution:
  def maximumOr(self, nums: List[int], k: int) -> int:
    n = len(nums)
    # prefix[i] := nums[0] | nums[1] | ... | nums[i - 1]
    prefix = [0] * n
    # suffix[i] := nums[i + 1] | nums[i + 2] | ... nums[n - 1]
    suffix = [0] * n

    for i in range(1, n):
      prefix[i] = prefix[i - 1] | nums[i - 1]

    for i in range(n - 2, -1, -1):
      suffix[i] = suffix[i + 1] | nums[i + 1]

    # For each num, greedily shift it left by k bits.
    return max(p | num << k | s for num, p, s in zip(nums, prefix, suffix))


# Link: https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/description/
class Solution:
  def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
    n = len(A)
    prefixCommon = 0
    ans = []
    count = [0] * (n + 1)

    for a, b in zip(A, B):
      count[a] += 1
      if count[a] == 2:
        prefixCommon += 1
      count[b] += 1
      if count[b] == 2:
        prefixCommon += 1
      ans.append(prefixCommon)

    return ans


# Link: https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/
class Solution:
  def deleteDuplicates(self, head: ListNode) -> ListNode:
    dummy = ListNode(0, head)
    prev = dummy

    while head:
      while head.next and head.val == head.next.val:
        head = head.next
      if prev.next == head:
        prev = prev.next
      else:
        prev.next = head.next
      head = head.next

    return dummy.next


# Link: https://leetcode.com/problems/convert-an-array-into-a-2d-array-with-conditions/description/
class Solution:
  def findMatrix(self, nums: List[int]) -> List[List[int]]:
    # The number of rows we need equals the maximum frequency.
    ans = []
    count = [0] * (len(nums) + 1)

    for num in nums:
      count[num] += 1
      # Construct `ans` on demand.
      if count[num] > len(ans):
        ans.append([])
      ans[count[num] - 1].append(num)

    return ans


# Link: https://leetcode.com/problems/append-k-integers-with-minimal-sum/description/
class Solution:
  def minimalKSum(self, nums: List[int], k: int) -> int:
    ans = 0
    nums.append(0)
    nums.sort()

    for a, b in zip(nums, nums[1:]):
      if a == b:
        continue
      l = a + 1
      r = min(a + k, b - 1)
      ans += (l + r) * (r - l + 1) // 2
      k -= r - l + 1
      if k == 0:
        return ans

    if k > 0:
      l = nums[-1] + 1
      r = nums[-1] + k
      ans += (l + r) * (r - l + 1) // 2

    return ans


# Link: https://leetcode.com/problems/decrease-elements-to-make-array-zigzag/description/
class Solution:
  def movesToMakeZigzag(self, nums: List[int]) -> int:
    decreasing = [0] * 2

    for i, num in enumerate(nums):
      l = nums[i - 1] if i > 0 else 1001
      r = nums[i + 1] if i + 1 < len(nums) else 1001
      decreasing[i % 2] += max(0, num - min(l, r) + 1)

    return min(decreasing[0], decreasing[1])


# Link: https://leetcode.com/problems/maximum-points-in-an-archery-competition/description/
class Solution:
  def maximumBobPoints(self, numArrows: int, aliceArrows: List[int]) -> List[int]:
    allMask = (1 << 12) - 1
    maxPoint = 0
    maxMask = 0

    def getShotableAndPoint(mask: int, leftArrows: int) -> Tuple[bool, int]:
      point = 0
      for i in range(12):
        if mask >> i & 1:
          leftArrows -= aliceArrows[i] + 1
          point += i
      return leftArrows >= 0, point

    for mask in range(allMask):
      shotable, point = getShotableAndPoint(mask, numArrows)
      if shotable and point > maxPoint:
        maxPoint = point
        maxMask = mask

    def getBobsArrows(mask: int, leftArrows: int) -> List[int]:
      bobsArrows = [0] * 12
      for i in range(12):
        if mask >> i & 1:
          bobsArrows[i] = aliceArrows[i] + 1
          leftArrows -= aliceArrows[i] + 1
      bobsArrows[0] = leftArrows
      return bobsArrows

    return getBobsArrows(maxMask, numArrows)


# Link: https://leetcode.com/problems/maximum-length-of-repeated-subarray/description/
class Solution:
  def findLength(self, nums1: List[int], nums2: List[int]) -> int:
    m = len(nums1)
    n = len(nums2)
    ans = 0
    # dp[i][j] := the maximum length of a subarray that appears in both
    # nums1[i..m) and nums2[j..n)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in reversed(range(m)):
      for j in reversed(range(n)):
        if nums1[i] == nums2[j]:
          dp[i][j] = dp[i + 1][j + 1] + 1
          ans = max(ans, dp[i][j])

    return ans


# Link: https://leetcode.com/problems/maximum-length-of-repeated-subarray/description/
class Solution:
  def findLength(self, nums1: List[int], nums2: List[int]) -> int:
    ans = 0
    dp = [0] * (len(nums2) + 1)

    for a in reversed(nums1):
      for j, b in enumerate(nums2):  # The order is important.
        dp[j] = dp[j + 1] + 1 if a == b else 0
        ans = max(ans, dp[j])

    return ans


# Link: https://leetcode.com/problems/find-score-of-an-array-after-marking-all-elements/description/
class Solution:
  def findScore(self, nums: List[int]) -> int:
    ans = 0
    seen = set()

    for num, i in sorted([(num, i) for i, num in enumerate(nums)]):
      if i in seen:
        continue
      seen.add(i - 1)
      seen.add(i + 1)
      seen.add(i)
      ans += num

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-set-cooking-time/description/
class Solution:
  def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
    ans = math.inf
    mins = 99 if targetSeconds > 5999 else targetSeconds // 60
    secs = targetSeconds - mins * 60

    def getCost(mins: int, secs: int) -> int:
      cost = 0
      curr = str(startAt)
      for c in str(mins * 100 + secs):
        if c == curr:
          cost += pushCost
        else:
          cost += moveCost + pushCost
          curr = c
      return cost

    while secs < 100:
      ans = min(ans, getCost(mins, secs))
      mins -= 1
      secs += 60

    return ans


# Link: https://leetcode.com/problems/time-needed-to-rearrange-a-binary-string/description/
class Solution:
  def secondsToRemoveOccurrences(self, s: str) -> int:
    ans = 0
    zeros = 0

    for c in s:
      if c == '0':
        zeros += 1
      elif zeros > 0:  # c == '1'
        ans = max(ans + 1, zeros)

    return ans


# Link: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
class Solution:
  def searchRange(self, nums: List[int], target: int) -> List[int]:
    l = bisect_left(nums, target)
    if l == len(nums) or nums[l] != target:
      return -1, -1
    r = bisect_right(nums, target) - 1
    return l, r


# Link: https://leetcode.com/problems/next-permutation/description/
class Solution:
  def nextPermutation(self, nums: List[int]) -> None:
    n = len(nums)

    # From back to front, find the first number < nums[i + 1].
    i = n - 2
    while i >= 0:
      if nums[i] < nums[i + 1]:
        break
      i -= 1

    # From back to front, find the first number > nums[i], swap it with nums[i].
    if i >= 0:
      for j in range(n - 1, i, -1):
        if nums[j] > nums[i]:
          nums[i], nums[j] = nums[j], nums[i]
          break

    def reverse(nums: List[int], l: int, r: int) -> None:
      while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1

    # Reverse nums[i + 1..n - 1].
    reverse(nums, i + 1, len(nums) - 1)


# Link: https://leetcode.com/problems/the-k-strongest-values-in-an-array/description/
class Solution:
  def getStrongest(self, arr: List[int], k: int) -> List[int]:
    arr.sort()

    ans = []
    median = arr[(len(arr) - 1) // 2]
    l = 0
    r = len(arr) - 1

    for _ in range(k):
      if median - arr[l] > arr[r] - median:
        ans.append(arr[l])
        l -= 1
      else:
        ans.append(arr[r])
        r += 1

    return ans


# Link: https://leetcode.com/problems/design-browser-history/description/
class BrowserHistory:
  def __init__(self, homepage: str):
    self.history = []
    self.visit(homepage)

  def visit(self, url: str) -> None:
    self.history.append(url)
    self.future = []

  def back(self, steps: int) -> str:
    while len(self.history) > 1 and steps > 0:
      self.future.append(self.history.pop())
      steps -= 1
    return self.history[-1]

  def forward(self, steps: int) -> str:
    while self.future and steps > 0:
      self.history.append(self.future.pop())
      steps -= 1
    return self.history[-1]


# Link: https://leetcode.com/problems/design-browser-history/description/
class Node:
  def __init__(self, url: str):
    self.prev = None
    self.next = None
    self.url = url


class BrowserHistory:
  def __init__(self, homepage: str):
    self.curr = Node(homepage)

  def visit(self, url: str) -> None:
    self.curr.next = Node(url)
    self.curr.next.prev = self.curr
    self.curr = self.curr.next

  def back(self, steps: int) -> str:
    while self.curr.prev and steps > 0:
      self.curr = self.curr.prev
      steps -= 1
    return self.curr.url

  def forward(self, steps: int) -> str:
    while self.curr.next and steps > 0:
      self.curr = self.curr.next
      steps -= 1
    return self.curr.url


# Link: https://leetcode.com/problems/design-browser-history/description/
class BrowserHistory:
  def __init__(self, homepage: str):
    self.urls = []
    self.index = -1
    self.lastIndex = -1
    self.visit(homepage)

  def visit(self, url: str) -> None:
    self.index += 1
    if self.index < len(self.urls):
      self.urls[self.index] = url
    else:
      self.urls.append(url)
    self.lastIndex = self.index

  def back(self, steps: int) -> str:
    self.index = max(0, self.index - steps)
    return self.urls[self.index]

  def forward(self, steps: int) -> str:
    self.index = min(self.lastIndex, self.index + steps)
    return self.urls[self.index]


# Link: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/
class Solution:
  def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    ans = 0
    distinct = 0
    count = collections.Counter()

    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      if count[c] == 1:
        distinct += 1
      while distinct == k + 1:
        count[s[l]] -= 1
        if count[s[l]] == 0:
          distinct -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/description/
class Solution:
  def minRemoveToMakeValid(self, s: str) -> str:
    stack = []  # unpaired '(' indices
    chars = [c for c in s]

    for i, c in enumerate(chars):
      if c == '(':
        stack.append(i)  # Record the unpaired '(' index.
      elif c == ')':
        if stack:
          stack.pop()  # Find a pair
        else:
          chars[i] = '*'  # Mark the unpaired ')' as '*'.

    # Mark the unpaired '(' as '*'.
    while stack:
      chars[stack.pop()] = '*'

    return ''.join(chars).replace('*', '')


# Link: https://leetcode.com/problems/first-day-where-you-have-been-in-all-the-rooms/description/
class Solution:
  def firstDayBeenInAllRooms(self, nextVisit: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nextVisit)
    # dp[i] := the number of days to visit room i for the first time
    dp = [0] * n

    # Whenever we visit i, visit times of room[0..i - 1] are all even.
    # Therefore, the rooms before i can be seen as reset and we can safely
    # reuse dp[0..i - 1] as first-time visit to get second-time visit.
    for i in range(1, n):
      # The total days to visit room[i] is the sum of
      #   * dp[i - 1]: 1st-time visit room[i - 1]
      #   * 1: visit room[nextVisit[i - 1]]
      #   * dp[i - 1] - dp[nextVisit[i - 1]]: 2-time visit room[i - 1]
      #   * 1: visit room[i]
      dp[i] = (2 * dp[i - 1] - dp[nextVisit[i - 1]] + 2) % kMod

    return dp[-1]


# Link: https://leetcode.com/problems/minimum-number-of-changes-to-make-binary-string-beautiful/description/
class Solution:
  def minChanges(self, s: str) -> int:
    return sum(a != b for a, b in zip(s[::2], s[1::2]))


# Link: https://leetcode.com/problems/magnetic-force-between-two-balls/description/
class Solution:
  def maxDistance(self, position: List[int], m: int) -> int:
    position.sort()

    l = 1
    r = position[-1] - position[0]

    def numBalls(force: int) -> int:
      balls = 0
      prevPosition = -force
      for pos in position:
        if pos - prevPosition >= force:
          balls += 1
          prevPosition = pos
      return balls

    while l < r:
      mid = r - (r - l) // 2
      if numBalls(mid) >= m:
        l = mid
      else:
        r = mid - 1

    return l


# Link: https://leetcode.com/problems/fraction-addition-and-subtraction/description/
class Solution:
  def fractionAddition(self, expression: str) -> str:
    ints = list(map(int, re.findall('[+-]?[0-9]+', expression)))
    A = 0
    B = 1

    for a, b in zip(ints[::2], ints[1::2]):
      A = A * b + a * B
      B *= b
      g = math.gcd(A, B)
      A //= g
      B //= g

    return str(A) + '/' + str(B)


# Link: https://leetcode.com/problems/finding-pairs-with-a-certain-sum/description/
class FindSumPairs:
  def __init__(self, nums1: List[int], nums2: List[int]):
    self.nums1 = nums1
    self.nums2 = nums2
    self.count2 = collections.Counter(nums2)

  def add(self, index: int, val: int) -> None:
    self.count2[self.nums2[index]] -= 1
    self.nums2[index] += val
    self.count2[self.nums2[index]] += 1

  def count(self, tot: int) -> int:
    ans = 0
    for num in self.nums1:
      ans += self.count2[tot - num]
    return ans


# Link: https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/description/
class Solution:
  def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
    for num in range(1, 7):
      if all(num in pair for pair in zip(tops, bottoms)):
        return len(tops) - max(tops.count(num), bottoms.count(num))
    return -1


# Link: https://leetcode.com/problems/single-element-in-a-sorted-array/description/
class Solution:
  def singleNonDuplicate(self, nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1

    while l < r:
      m = (l + r) // 2
      if m % 2 == 1:
        m -= 1
      if nums[m] == nums[m + 1]:
        l = m + 2
      else:
        r = m

    return nums[l]


# Link: https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/
class Solution:
  def minIncrementForUnique(self, nums: List[int]) -> int:
    ans = 0
    minAvailable = 0

    for num in sorted(nums):
      ans += max(minAvailable - num, 0)
      minAvailable = max(minAvailable, num) + 1

    return ans


# Link: https://leetcode.com/problems/water-and-jug-problem/description/
class Solution:
  def canMeasureWater(self, jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
    return targetCapacity == 0 or \
        jug1Capacity + jug2Capacity >= targetCapacity and \
        targetCapacity % gcd(jug1Capacity, jug2Capacity) == 0


# Link: https://leetcode.com/problems/minimum-absolute-difference-queries/description/
class Solution:
  def minDifference(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    numToIndices = [[] for _ in range(101)]

    for i, num in enumerate(nums):
      numToIndices[num].append(i)

    if len(numToIndices[nums[0]]) == len(nums):
      return [-1] * len(queries)

    ans = []

    for l, r in queries:
      prevNum = -1
      minDiff = 101
      for num in range(1, 101):
        indices = numToIndices[num]
        i = bisect_left(indices, l)
        if i == len(indices) or indices[i] > r:
          continue
        if prevNum != -1:
          minDiff = min(minDiff, num - prevNum)
        prevNum = num
      ans.append(-1 if minDiff == 101 else minDiff)

    return ans


# Link: https://leetcode.com/problems/maximum-total-importance-of-roads/description/
class Solution:
  def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
    count = [0] * n

    for u, v in roads:
      count[u] += 1
      count[v] += 1

    count.sort()
    return sum((i + 1) * c for i, c in enumerate(count))


# Link: https://leetcode.com/problems/score-of-parentheses/description/
class Solution:
  def scoreOfParentheses(self, s: str) -> int:
    ans = 0
    layer = 0

    for a, b in itertools.pairwise(s):
      if a + b == '()':
        ans += 1 << layer
      layer += 1 if a == '(' else -1

    return ans


# Link: https://leetcode.com/problems/linked-list-cycle-ii/description/
class Solution:
  def detectCycle(self, head: ListNode) -> ListNode:
    slow = head
    fast = head

    while fast and fast.next:
      slow = slow.next
      fast = fast.next.next
      if slow == fast:
        slow = head
        while slow != fast:
          slow = slow.next
          fast = fast.next
        return slow

    return None


# Link: https://leetcode.com/problems/out-of-boundary-paths/description/
class Solution:
  def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    kMod = 1_000_000_007
    ans = 0
    # dp[i][j] := the number of paths to move the ball (i, j) out-of-bounds
    dp = [[0] * n for _ in range(m)]
    dp[startRow][startColumn] = 1

    for _ in range(maxMove):
      newDp = [[0] * n for _ in range(m)]
      for i in range(m):
        for j in range(n):
          if dp[i][j] > 0:
            for dx, dy in dirs:
              x = i + dx
              y = j + dy
              if x < 0 or x == m or y < 0 or y == n:
                ans = (ans + dp[i][j]) % kMod
              else:
                newDp[x][y] = (newDp[x][y] + dp[i][j]) % kMod
      dp = newDp

    return ans


# Link: https://leetcode.com/problems/out-of-boundary-paths/description/
class Solution:
  def findPaths(self, m, n, maxMove, startRow, startColumn):
    kMod = 1000000007

    @functools.lru_cache(None)
    def dp(k: int, i: int, j: int) -> int:
      """
      Returns the number of paths to move the ball at (i, j) out-of-bounds with
      k moves.
      """
      if i < 0 or i == m or j < 0 or j == n:
        return 1
      if k == 0:
        return 0
      return (dp(k - 1, i + 1, j) + dp(k - 1, i - 1, j) +
              dp(k - 1, i, j + 1) + dp(k - 1, i, j - 1)) % kMod

    return dp(maxMove, startRow, startColumn)


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/description/
class Solution:
  # Same as 160. Intersection of Two Linked Lists
  def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
    a = p
    b = q

    while a != b:
      a = a.parent if a else q
      b = b.parent if b else p

    return a


# Link: https://leetcode.com/problems/count-artifacts-that-can-be-extracted/description/
class Solution:
  def digArtifacts(self, n: int, artifacts: List[List[int]], dig: List[List[int]]) -> int:
    digged = set((r, c) for r, c in dig)

    def canExtract(a: List[int]) -> bool:
      for i in range(a[0], a[2] + 1):
        for j in range(a[1], a[3] + 1):
          if (i, j) not in digged:
            return False
      return True

    return sum(canExtract(a) for a in artifacts)


# Link: https://leetcode.com/problems/minimum-moves-to-reach-target-score/description/
class Solution:
  def minMoves(self, target: int, maxDoubles: int) -> int:
    steps = 0

    while target > 1 and maxDoubles:
      if target & 1:
        target -= 1
      else:
        target //= 2
        maxDoubles -= 1
      steps += 1

    return steps + target - 1


# Link: https://leetcode.com/problems/number-of-unique-categories/description/
# Definition for a category handler.
# class CategoryHandler:
#   def haveSameCategory(self, a: int, b: int) -> bool:
#     pass

class Solution:
  def numberOfCategories(self, n: int, categoryHandler: Optional['CategoryHandler']) -> int:
    ans = 0

    for i in range(n):
      if not any(categoryHandler.haveSameCategory(i, j) for j in range(i)):
        ans += 1

    return ans


# Link: https://leetcode.com/problems/maximum-erasure-value/description/
class Solution:
  def maximumUniqueSubarray(self, nums: List[int]) -> int:
    ans = 0
    score = 0
    seen = set()

    l = 0
    for r, num in enumerate(nums):
      while num in seen:
        score -= nums[l]
        seen.remove(nums[l])
        l += 1
      seen.add(nums[r])
      score += nums[r]
      ans = max(ans, score)

    return ans


# Link: https://leetcode.com/problems/kth-largest-sum-in-a-binary-tree/description/
class Solution:
  def kthLargestLevelSum(self, root: Optional[TreeNode], k: int) -> int:
    levelSums = []

    def dfs(root: Optional[TreeNode], level: int) -> None:
      if not root:
        return
      if len(levelSums) == level:
        levelSums.append(0)
      levelSums[level] += root.val
      dfs(root.left, level + 1)
      dfs(root.right, level + 1)

    dfs(root, 0)
    if len(levelSums) < k:
      return -1

    return sorted(levelSums, reverse=True)[k - 1]


# Link: https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
class Solution:
  def evalRPN(self, tokens: List[str]) -> int:
    stack = []
    op = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),
    }

    for token in tokens:
      if token in op:
        b = stack.pop()
        a = stack.pop()
        stack.append(op[token](a, b))
      else:
        stack.append(int(token))

    return stack.pop()


# Link: https://leetcode.com/problems/maximize-win-from-two-segments/description/
class Solution:
  def maximizeWin(self, prizePositions: List[int], k: int) -> int:
    ans = 0
    # dp[i] := the maximum number of prizes to choose the first i
    # `prizePositions`
    dp = [0] * (len(prizePositions) + 1)

    j = 0
    for i, prizePosition in enumerate(prizePositions):
      while prizePosition - prizePositions[j] > k:
        j += 1
      covered = i - j + 1
      dp[i + 1] = max(dp[i], covered)
      ans = max(ans, dp[j] + covered)

    return ans


# Link: https://leetcode.com/problems/max-increase-to-keep-city-skyline/description/
class Solution:
  def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    rowMax = list(map(max, grid))
    colMax = list(map(max, zip(*grid)))
    return sum(min(i, j) for i in rowMax for j in colMax) - sum(map(sum, grid))


# Link: https://leetcode.com/problems/sentence-similarity-iii/description/
class Solution:
  def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
    if len(sentence1) == len(sentence2):
      return sentence1 == sentence2

    words1 = sentence1.split()
    words2 = sentence2.split()
    m, n = map(len, (words1, words2))
    if m > n:
      return self.areSentencesSimilar(sentence2, sentence1)

    i = 0  # words1's index
    while i < m and words1[i] == words2[i]:
      i += 1
    while i < m and words1[i] == words2[i + n - m]:
      i += 1

    return i == m


# Link: https://leetcode.com/problems/rabbits-in-forest/description/
class Solution:
  def numRabbits(self, answers: List[int]) -> int:
    ans = 0
    count = collections.Counter()

    for answer in answers:
      if count[answer] % (answer + 1) == 0:
        ans += answer + 1
      count[answer] += 1

    return ans


# Link: https://leetcode.com/problems/number-of-subarrays-that-match-a-pattern-i/description/
class Solution:
  def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
    def getNum(a: int, b: int) -> int:
      if a < b:
        return 1
      if a > b:
        return -1
      return 0

    numsPattern = [getNum(a, b) for a, b in itertools.pairwise(nums)]
    return self._kmp(numsPattern, pattern)

  def _kmp(self, nums: List[int], pattern: List[int]) -> int:
    """Returns the number of occurrences of the pattern in `nums`."""

    def getLPS(nums: List[int]) -> List[int]:
      """
      Returns the lps array, where lps[i] is the length of the longest prefix of
      nums[0..i] which is also a suffix of this substring.
      """
      lps = [0] * len(nums)
      j = 0
      for i in range(1, len(nums)):
        while j > 0 and nums[j] != nums[i]:
          j = lps[j - 1]
        if nums[i] == nums[j]:
          lps[i] = j + 1
          j += 1
      return lps

    lps = getLPS(pattern)
    res = 0
    i = 0  # s' index
    j = 0  # pattern's index
    while i < len(nums):
      if nums[i] == pattern[j]:
        i += 1
        j += 1
        if j == len(pattern):
          res += 1
          j = lps[j - 1]
      # Mismatch after j matches.
      elif j != 0:
          # Don't match lps[0..lps[j - 1]] since they will match anyway.
        j = lps[j - 1]
      else:
        i += 1
    return res


# Link: https://leetcode.com/problems/find-good-days-to-rob-the-bank/description/
class Solution:
  def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
    n = len(security)
    dec = [0] * n  # dec[i] := the number of continuous decreasing numbers before i
    inc = [0] * n  # inc[i] := the number of continuous increasing numbers after i

    for i in range(1, n):
      if security[i - 1] >= security[i]:
        dec[i] = dec[i - 1] + 1

    for i in range(n - 2, -1, -1):
      if security[i] <= security[i + 1]:
        inc[i] = inc[i + 1] + 1

    return [i for i, (a, b) in enumerate(zip(dec, inc))
            if a >= time and b >= time]


# Link: https://leetcode.com/problems/count-number-of-homogenous-substrings/description/
class Solution:
  def countHomogenous(self, s: str) -> int:
    kMod = 1_000_000_007
    ans = 0
    count = 0
    currentChar = '@'

    for c in s:
      count = count + 1 if c == currentChar else 1
      currentChar = c
      ans += count
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/bulb-switcher-ii/description/
class Solution:
  def flipLights(self, n: int, m: int) -> int:
    n = min(n, 3)

    if m == 0:
      return 1
    if m == 1:
      return [2, 3, 4][n - 1]
    if m == 2:
      return [2, 4, 7][n - 1]

    return [2, 4, 8][n - 1]


# Link: https://leetcode.com/problems/the-latest-time-to-catch-a-bus/description/
class Solution:
  def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
    buses.sort()
    passengers.sort()

    if passengers[0] > buses[-1]:
      return buses[-1]

    ans = passengers[0] - 1
    i = 0  # buses' index
    j = 0  # passengers' index

    while i < len(buses):
      # Greedily make passengers catch `buses[i]`.
      arrived = 0
      while arrived < capacity and j < len(passengers) and passengers[j] <= buses[i]:
        if j > 0 and passengers[j] != passengers[j - 1] + 1:
          ans = passengers[j] - 1
        j += 1
        arrived += 1
      # There's room for `buses[i]` to carry a passenger arriving at the
      # `buses[i]`.
      if arrived < capacity and j > 0 and passengers[j - 1] != buses[i]:
        ans = buses[i]
      i += 1

    return ans


# Link: https://leetcode.com/problems/sequence-reconstruction/description/
class Solution:
  def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:
    if not seqs:
      return False

    n = len(org)
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n

    # Build the graph.
    for seq in seqs:
      if len(seq) == 1 and seq[0] < 1 or seq[0] > n:
        return False
      for u, v in zip(seq, seq[1:]):
        if u < 1 or u > n or v < 1 or v > n:
          return False
        graph[u - 1].append(v - 1)
        inDegrees[v - 1] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])
    i = 0  # org's index

    while q:
      if len(q) > 1:
        return False
      u = q.popleft()
      if u != org[i] - 1:
        return False
      i += 1
      for v in graph[u]:
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return i == n


# Link: https://leetcode.com/problems/reduce-array-size-to-the-half/description/
class Solution:
  def minSetSize(self, arr: List[int]) -> int:
    n = len(arr)

    count = collections.Counter(arr).most_common()
    count.sort(key=lambda c: -c[1])

    summ = 0
    for i, c in enumerate(count):
      summ += c[1]
      if summ >= n // 2:
        return i + 1


# Link: https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/description/
class Solution:
  def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
    ans = 0
    l = -1
    r = -1

    for i, num in enumerate(nums):
      if num > right:  # Handle the reset value.
        l = i
      if num >= left:  # Handle the reset and the needed value.
        r = i
      ans += r - l

    return ans


# Link: https://leetcode.com/problems/sum-of-remoteness-of-all-cells/description/
class Solution:
  def sumRemoteness(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    n = len(grid)
    summ = sum(max(0, cell) for row in grid for cell in row)
    ans = 0

    def dfs(i: int, j: int) -> Tuple[int, int]:
      """
      Returns the (count, componentSum) of the connected component that contains
      (x, y).
      """
      if i < 0 or i == len(grid) or j < 0 or j == len(grid[0]):
        return (0, 0)
      if grid[i][j] == -1:
        return (0, 0)

      count = 1
      componentSum = grid[i][j]
      grid[i][j] = -1  # Mark as visited.

      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        nextCount, nextComponentSum = dfs(x, y)
        count += nextCount
        componentSum += nextComponentSum

      return (count, componentSum)

    for i in range(n):
      for j in range(n):
        if grid[i][j] > 0:
          count, componentSum = dfs(i, j)
          ans += (summ - componentSum) * count

    return ans


# Link: https://leetcode.com/problems/design-bounded-blocking-queue/description/
from threading import Semaphore


class BoundedBlockingQueue:
  def __init__(self, capacity: int):
    self.q = collections.deque()
    self.enqueueSemaphore = Semaphore(capacity)
    self.dequeueSemaphore = Semaphore(0)

  def enqueue(self, element: int) -> None:
    self.enqueueSemaphore.acquire()
    self.q.append(element)
    self.dequeueSemaphore.release()

  def dequeue(self) -> int:
    self.dequeueSemaphore.acquire()
    element = self.q.popleft()
    self.enqueueSemaphore.release()
    return element

  def size(self) -> int:
    return len(self.q)


# Link: https://leetcode.com/problems/find-the-duplicate-number/description/
class Solution:
  def findDuplicate(self, nums: List[int]) -> int:
    slow = nums[nums[0]]
    fast = nums[nums[nums[0]]]

    while slow != fast:
      slow = nums[slow]
      fast = nums[nums[fast]]

    slow = nums[0]

    while slow != fast:
      slow = nums[slow]
      fast = nums[fast]

    return slow


# Link: https://leetcode.com/problems/execution-of-all-suffix-instructions-staying-in-a-grid/description/
class Solution:
  def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    m = len(s)
    uMost = startPos[0] + 1
    dMost = n - startPos[0]
    lMost = startPos[1] + 1
    rMost = n - startPos[1]

    ans = [0] * m
    reach = {(0, None): m, (None, 0): m}
    x = 0
    y = 0

    for i in reversed(range(m)):
      dx, dy = moves[s[i]]
      x -= dx
      y -= dy
      reach[(x, None)] = i
      reach[(None, y)] = i
      out = min(reach.get((x - uMost, None), math.inf),
                reach.get((x + dMost, None), math.inf),
                reach.get((None, y - lMost), math.inf),
                reach.get((None, y + rMost), math.inf))
      ans[i] = m - i if out == math.inf else out - i - 1

    return ans


# Link: https://leetcode.com/problems/difference-of-number-of-distinct-values-on-diagonals/description/
class Solution:
  def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
    m = len(grid)
    n = len(grid[0])
    ans = [[0] * n for _ in range(m)]

    def fillInDiagonal(i: int, j: int) -> None:
      topLeft = set()
      bottomRight = set()

      # Fill in the diagonal from the top-left to the bottom-right.
      while i < len(grid) and j < len(grid[0]):
        ans[i][j] = len(topLeft)
        # Post-addition, so this information can be utilized in subsequent cells.
        topLeft.add(grid[i][j])
        i += 1
        j += 1

      i -= 1
      j -= 1

      # Fill in the diagonal from the bottom-right to the top-left.
      while i >= 0 and j >= 0:
        ans[i][j] = abs(ans[i][j] - len(bottomRight))
        # Post-addition, so this information can be utilized in subsequent cells.
        bottomRight.add(grid[i][j])
        i -= 1
        j -= 1

    for i in range(m):
      fillInDiagonal(i, 0)

    for j in range(1, n):
      fillInDiagonal(0, j)

    return ans


# Link: https://leetcode.com/problems/graph-valid-tree/description/
class Solution:
  def validTree(self, n: int, edges: List[List[int]]) -> bool:
    if n == 0 or len(edges) != n - 1:
      return False

    graph = [[] for _ in range(n)]
    q = collections.deque([0])
    seen = {0}

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    while q:
      u = q.popleft()
      for v in graph[u]:
        if v not in seen:
          q.append(v)
          seen.add(v)

    return len(seen) == n


# Link: https://leetcode.com/problems/graph-valid-tree/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1
    self.count -= 1

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def validTree(self, n: int, edges: List[List[int]]) -> bool:
    if n == 0 or len(edges) != n - 1:
      return False

    uf = UnionFind(n)

    for u, v in edges:
      uf.unionByRank(u, v)

    return uf.count == 1


# Link: https://leetcode.com/problems/maximum-product-of-word-lengths/description/
class Solution:
  def maxProduct(self, words: List[str]) -> int:
    ans = 0

    def getMask(word: str) -> int:
      mask = 0
      for c in word:
        mask |= 1 << ord(c) - ord('a')
      return mask

    masks = [getMask(word) for word in words]

    for i in range(len(words)):
      for j in range(i):
        if not (masks[i] & masks[j]):
          ans = max(ans, len(words[i]) * len(words[j]))

    return ans


# Link: https://leetcode.com/problems/number-of-zero-filled-subarrays/description/
class Solution:
  def zeroFilledSubarray(self, nums: List[int]) -> int:
    ans = 0
    indexBeforeZero = -1

    for i, num in enumerate(nums):
      if num:
        indexBeforeZero = i
      else:
        ans += i - indexBeforeZero

    return ans


# Link: https://leetcode.com/problems/print-zero-even-odd/description/
from threading import Semaphore


class ZeroEvenOdd:
  def __init__(self, n):
    self.n = n
    self.zeroSemaphore = Semaphore(1)
    self.evenSemaphore = Semaphore(0)
    self.oddSemaphore = Semaphore(0)

  # printNumber(x) outputs "x", where x is an integer.
  def zero(self, printNumber: 'Callable[[int], None]') -> None:
    for i in range(self.n):
      self.zeroSemaphore.acquire()
      printNumber(0)
      (self.oddSemaphore if i & 2 == 0 else self.evenSemaphore).release()

  def even(self, printNumber: 'Callable[[int], None]') -> None:
    for i in range(2, self.n + 1, 2):
      self.evenSemaphore.acquire()
      printNumber(i)
      self.zeroSemaphore.release()

  def odd(self, printNumber: 'Callable[[int], None]') -> None:
    for i in range(1, self.n + 1, 2):
      self.oddSemaphore.acquire()
      printNumber(i)
      self.zeroSemaphore.release()


# Link: https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/description/
class Solution:
  def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
    ans = 0
    tree = [[] for _ in range(len(roads) + 1)]

    for u, v in roads:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int, prev: int) -> int:
      nonlocal ans
      people = 1
      for v in tree[u]:
        if v == prev:
          continue
        people += dfs(v, u)
      if u > 0:
        # the number of cars needed
        ans += int(math.ceil(people / seats))
      return people

    dfs(0, -1)
    return ans


# Link: https://leetcode.com/problems/h-index/description/
class Solution:
  def hIndex(self, citations: List[int]) -> int:
    n = len(citations)

    citations.sort()

    for i, citation in enumerate(citations):
      if citation >= n - i:
        return n - i

    return 0


# Link: https://leetcode.com/problems/h-index/description/
class Solution:
  def hIndex(self, citations: List[int]) -> int:
    n = len(citations)
    accumulate = 0
    count = [0] * (n + 1)

    for citation in citations:
      count[min(citation, n)] += 1

    # To find the maximum h-index, loop from the back to the front.
    # i := the candidate's h-index
    for i, c in reversed(list(enumerate(count))):
      accumulate += c
      if accumulate >= i:
        return i


# Link: https://leetcode.com/problems/number-of-unique-flavors-after-sharing-k-candies/description/
class Solution:
  def shareCandies(self, candies: List[int], k: int) -> int:
    ans = 0
    count = collections.Counter(candies)
    unique = len(count)

    for i, candy in enumerate(candies):
      count[candy] -= 1
      if count[candy] == 0:
        del count[candy]
        unique -= 1
      if i >= k:
        count[candies[i - k]] += 1
        if count[candies[i - k]] == 1:
          unique += 1
      if i >= k - 1:
        ans = max(ans, unique)

    return ans


# Link: https://leetcode.com/problems/unique-substrings-with-equal-digit-frequency/description/
class Solution:
  def equalDigitFrequency(self, s: str) -> int:
    power = 11
    kMod = 1_000_000_007
    seen = set()

    def isUnique(s: str, i: int, j: int) -> bool:
      count = [0] * 10
      unique = 0
      for k in range(i, j + 1):
        count[ord(s[k]) - ord('0')] += 1
        if count[ord(s[k]) - ord('0')] == 1:
          unique += 1
      maxCount = max(count)
      return maxCount * unique == j - i + 1

    def getRollingHash(s: str, i: int, j: int) -> int:
      hash = 0
      for k in range(i, j + 1):
        hash = (hash * power + val(s[k])) % kMod
      return hash

    def val(c: str) -> int:
      return ord(c) - ord('0') + 1

    for i in range(len(s)):
      for j in range(i, len(s)):
        if isUnique(s, i, j):
          seen.add(getRollingHash(s, i, j))

    return len(seen)


# Link: https://leetcode.com/problems/remove-duplicate-letters/description/
class Solution:
  def removeDuplicateLetters(self, s: str) -> str:
    ans = []
    count = collections.Counter(s)
    used = [False] * 26

    for c in s:
      count[c] -= 1
      if used[ord(c) - ord('a')]:
        continue
      while ans and ans[-1] > c and count[ans[-1]] > 0:
        used[ord(ans[-1]) - ord('a')] = False
        ans.pop()
      ans.append(c)
      used[ord(ans[-1]) - ord('a')] = True

    return ''.join(ans)


# Link: https://leetcode.com/problems/word-search/description/
class Solution:
  def exist(self, board: List[List[str]], word: str) -> bool:
    m = len(board)
    n = len(board[0])

    def dfs(i: int, j: int, s: int) -> bool:
      if i < 0 or i == m or j < 0 or j == n:
        return False
      if board[i][j] != word[s] or board[i][j] == '*':
        return False
      if s == len(word) - 1:
        return True

      cache = board[i][j]
      board[i][j] = '*'
      isExist = \
          dfs(i + 1, j, s + 1) or \
          dfs(i - 1, j, s + 1) or \
          dfs(i, j + 1, s + 1) or \
          dfs(i, j - 1, s + 1)
      board[i][j] = cache

      return isExist

    return any(dfs(i, j, 0) for i in range(m) for j in range(n))


# Link: https://leetcode.com/problems/minimum-split-into-subarrays-with-gcd-greater-than-one/description/
class Solution:
  def minimumSplits(self, nums: List[int]) -> int:
    ans = 1
    gcd = nums[0]

    for num in nums:
      newGcd = math.gcd(gcd, num)
      if newGcd > 1:
        gcd = newGcd
      else:
        gcd = num
        ans += 1

    return ans


# Link: https://leetcode.com/problems/coin-change/description/
class Solution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    # dp[i] := the minimum number Of coins to make up i
    dp = [0] + [amount + 1] * amount

    for coin in coins:
      for i in range(coin, amount + 1):
        dp[i] = min(dp[i], dp[i - coin] + 1)

    return -1 if dp[amount] == amount + 1 else dp[amount]


# Link: https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/description/
class Solution:
  def minFlips(self, s: str) -> int:
    n = len(s)
    # count[0][0] :=  the number of '0' in the even indices
    # count[0][1] :=  the number of '0' in the odd indices
    # count[1][0] :=  the number of '1' in the even indices
    # count[1][1] :=  the number of '1' in the odd indices
    count = [[0] * 2 for _ in range(2)]

    for i, c in enumerate(s):
      count[ord(c) - ord('0')][i % 2] += 1

    # min(make all 0s in the even indices + make all 1s in the odd indices,
    #     make all 1s in the even indices + make all 0s in the odd indices)
    ans = min(count[1][0] + count[0][1], count[0][0] + count[1][1])

    for i, c in enumerate(s):
      count[ord(c) - ord('0')][i % 2] -= 1
      count[ord(c) - ord('0')][(n + i) % 2] += 1
      ans = min(ans, count[1][0] + count[0][1], count[0][0] + count[1][1])

    return ans


# Link: https://leetcode.com/problems/partition-array-according-to-given-pivot/description/
class Solution:
  def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
    return [num for num in nums if num < pivot] + \
        [num for num in nums if num == pivot] + \
        [num for num in nums if num > pivot]


# Link: https://leetcode.com/problems/reveal-cards-in-increasing-order/description/
class Solution:
  def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
    dq = collections.deque()

    for card in reversed(sorted(deck)):
      dq.rotate()
      dq.appendleft(card)

    return list(dq)


# Link: https://leetcode.com/problems/relocate-marbles/description/
class Solution:
  def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
    numsSet = set(nums)

    for f, t in zip(moveFrom, moveTo):
      numsSet.remove(f)
      numsSet.add(t)

    return sorted(numsSet)


# Link: https://leetcode.com/problems/binary-search-tree-iterator/description/
class BSTIterator:
  def __init__(self, root: Optional[TreeNode]):
    self.i = 0
    self.vals = []
    self._inorder(root)

  def next(self) -> int:
    self.i += 1
    return self.vals[self.i - 1]

  def hasNext(self) -> bool:
    return self.i < len(self.vals)

  def _inorder(self, root: Optional[TreeNode]) -> None:
    if not root:
      return
    self._inorder(root.left)
    self.vals.append(root.val)
    self._inorder(root.right)


# Link: https://leetcode.com/problems/binary-search-tree-iterator/description/
class BSTIterator:
  def __init__(self, root: Optional[TreeNode]):
    self.stack = []
    self._pushLeftsUntilNull(root)

  def next(self) -> int:
    root = self.stack.pop()
    self._pushLeftsUntilNull(root.right)
    return root.val

  def hasNext(self) -> bool:
    return self.stack

  def _pushLeftsUntilNull(self, root: Optional[TreeNode]) -> None:
    while root:
      self.stack.append(root)
      root = root.left


# Link: https://leetcode.com/problems/maximum-number-of-alloys/description/
class Solution:
  def maxNumberOfAlloys(self, n: int, k: int, budget: int,
                        composition: List[List[int]], stock: List[int],
                        costs: List[int]) -> int:
    l = 1
    r = 1_000_000_000

    def isPossible(m: int) -> bool:
      """Returns True if it's possible to create `m` alloys by using any machine."""
      # Try all the possible machines.
      for machine in composition:
        requiredMoney = 0
        for j in range(n):
          requiredUnits = max(0, machine[j] * m - stock[j])
          requiredMoney += requiredUnits * costs[j]
        if requiredMoney <= budget:
          return True
      return False

    while l < r:
      m = (l + r) // 2
      if isPossible(m):
        l = m + 1
      else:
        r = m

    return l - 1


# Link: https://leetcode.com/problems/maximum-compatibility-score-sum/description/
class Solution:
  def maxCompatibilitySum(self, students: List[List[int]], mentors: List[List[int]]) -> int:
    ans = 0

    def dfs(i: int, scoreSum: int, used: List[bool]) -> None:
      nonlocal ans
      if i == len(students):
        ans = max(ans, scoreSum)
        return

      for j, mentor in enumerate(mentors):
        if used[j]:
          continue
        used[j] = True  # The `mentors[j]` is used.
        dfs(i + 1, scoreSum + sum(s == m
                                  for s, m in zip(students[i], mentor)), used)
        used[j] = False

    dfs(0, 0, [False] * len(students))
    return ans


# Link: https://leetcode.com/problems/web-crawler/description/
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class HtmlParser(object):
#   def getUrls(self, url: str) -> List[str]:

class Solution:
  def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
    q = collections.deque([startUrl])
    seen = {startUrl}
    hostname = startUrl.split('/')[2]

    while q:
      currUrl = q.popleft()
      for url in htmlParser.getUrls(currUrl):
        if url in seen:
          continue
        if hostname in url:
          q.append(url)
          seen.add(url)

    return seen


# Link: https://leetcode.com/problems/solving-questions-with-brainpower/description/
class Solution:
  def mostPoints(self, questions: List[List[int]]) -> int:
    n = len(questions)
    # dp[i] := the maximum points starting from questions[i]
    dp = [0] * (n + 1)

    for i in reversed(range(n)):
      points, brainpower = questions[i]
      nextIndex = i + brainpower + 1
      nextPoints = dp[nextIndex] if nextIndex < n else 0
      dp[i] = max(points + nextPoints, dp[i + 1])

    return dp[0]


# Link: https://leetcode.com/problems/delete-operation-for-two-strings/description/
class Solution:
  def minDistance(self, word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    dp = [0] * (n + 1)

    for j in range(n + 1):
      dp[j] = j

    for i in range(1, m + 1):
      newDp = [i] + [0] * n
      for j in range(1, n + 1):
        if word1[i - 1] == word2[j - 1]:
          newDp[j] = dp[j - 1]
        else:
          newDp[j] = min(newDp[j - 1], dp[j]) + 1
      dp = newDp

    return dp[n]


# Link: https://leetcode.com/problems/count-the-hidden-sequences/description/
class Solution:
  def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
    prefix = 0
    mini = 0  # Starts from 0.
    maxi = 0  # Starts from 0.

    for d in differences:
      prefix += d
      mini = min(mini, prefix)
      maxi = max(maxi, prefix)

    return max(0, (upper - lower) - (maxi - mini) + 1)


# Link: https://leetcode.com/problems/count-the-hidden-sequences/description/
class Solution:
  def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
    prefix = [0] + list(itertools.accumulate(differences))
    return max(0, (upper - lower) - (max(prefix) - min(prefix)) + 1)


# Link: https://leetcode.com/problems/cinema-seat-allocation/description/
class Solution:
  def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
    ans = 0
    rowToSeats = collections.Counter()

    for row, seat in reservedSeats:
      rowToSeats[row] |= 1 << (seat - 1)

    for seats in rowToSeats.values():
      if (seats & 0b0111111110) == 0:
        # Can fit 2 four-person groups.
        ans += 2
      elif (seats & 0b0111100000) == 0 \
              or (seats & 0b0001111000) == 0 \
              or (seats & 0b0000011110) == 0:
        # Can fit 1 four-person group.
        ans += 1

    # Any empty row can fit 2 four-person groups.
    return ans + (n - len(rowToSeats)) * 2


# Link: https://leetcode.com/problems/find-players-with-zero-or-one-losses/description/
class Solution:
  def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
    ans = [[] for _ in range(2)]
    lossesCount = collections.Counter()

    for winner, loser in matches:
      if winner not in lossesCount:
        lossesCount[winner] = 0
      lossesCount[loser] += 1

    for player, nLosses in lossesCount.items():
      if nLosses < 2:
        ans[nLosses].append(player)

    return [sorted(ans[0]), sorted(ans[1])]


# Link: https://leetcode.com/problems/find-consecutive-integers-from-a-data-stream/description/
class DataStream:
  def __init__(self, value: int, k: int):
    self.value = value
    self.k = k
    self.q = deque()
    self.count = 0

  def consec(self, num: int) -> bool:
    if len(self.q) == self.k and self.q.popleft() == self.value:
      self.count -= 1
    if num == self.value:
      self.count += 1
    self.q.append(num)
    return self.count == self.k


# Link: https://leetcode.com/problems/encode-and-decode-strings/description/
class Codec:
  def encode(self, strs: List[str]) -> str:
    """Encodes a list of strings to a single string."""
    return ''.join(str(len(s)) + '/' + s for s in strs)

  def decode(self, s: str) -> List[str]:
    """Decodes a single string to a list of strings."""
    decoded = []

    i = 0
    while i < len(s):
      slash = s.find('/', i)
      length = int(s[i:slash])
      i = slash + length + 1
      decoded.append(s[slash + 1:i])

    return decoded


# Link: https://leetcode.com/problems/queries-on-a-permutation-with-key/description/
class FenwickTree:
  def __init__(self, n: int):
    self.sums = [0] * (n + 1)

  def update(self, i: int, delta: int) -> None:
    while i < len(self.sums):
      self.sums[i] += delta
      i += FenwickTree.lowbit(i)

  def get(self, i: int) -> int:
    summ = 0
    while i > 0:
      summ += self.sums[i]
      i -= FenwickTree.lowbit(i)
    return summ

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i


class Solution:
  def processQueries(self, queries: List[int], m: int) -> List[int]:
    ans = []
    # Map [-m, m] to [0, 2 * m].
    tree = FenwickTree(2 * m + 1)
    numToIndex = {num: num + m for num in range(1, m + 1)}

    for num in range(1, m + 1):
      tree.update(num + m, 1)

    nextEmptyIndex = m  # Map 0 to m.

    for query in queries:
      index = numToIndex[query]
      ans.append(tree.get(index - 1))
      # Move `query` from `index` to `nextEmptyIndex`.
      tree.update(index, -1)
      tree.update(nextEmptyIndex, 1)
      numToIndex[query] = nextEmptyIndex
      nextEmptyIndex -= 1

    return ans


# Link: https://leetcode.com/problems/paint-house/description/
class Solution:
  def minCost(self, costs: List[List[int]]) -> List[List[int]]:
    for i in range(1, len(costs)):
      costs[i][0] += min(costs[i - 1][1], costs[i - 1][2])
      costs[i][1] += min(costs[i - 1][0], costs[i - 1][2])
      costs[i][2] += min(costs[i - 1][0], costs[i - 1][1])

    return min(costs[-1])


# Link: https://leetcode.com/problems/4sum-ii/description/
class Solution:
  def fourSumCount(self, nums1: List[int], nums2: List[int],
                   nums3: List[int], nums4: List[int]) -> int:
    count = collections.Counter(a + b for a in nums1 for b in nums2)
    return sum(count[-c - d] for c in nums3 for d in nums4)


# Link: https://leetcode.com/problems/maximum-sum-of-almost-unique-subarray/description/
class Solution:
  def maxSum(self, nums: List[int], m: int, k: int) -> int:
    ans = 0
    summ = 0
    count = collections.Counter()

    for i, num in enumerate(nums):
      summ += num
      count[num] += 1
      if i >= k:
        numToRemove = nums[i - k]
        summ -= numToRemove
        count[numToRemove] -= 1
        if count[numToRemove] == 0:
          del count[numToRemove]
      if len(count) >= m:
        ans = max(ans, summ)

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/description/
class Solution:
  def maxNonOverlapping(self, nums: List[int], target: int) -> int:
    # Ending the subarray ASAP always has a better result.
    ans = 0
    prefix = 0
    prefixes = {0}

    # Greedily find the subarrays that equal to the target.
    for num in nums:
      # Check if there is a subarray ends in here and equals to the target.
      prefix += num
      if prefix - target in prefixes:
        # Find one and discard all the prefixes that have been used.
        ans += 1
        prefix = 0
        prefixes = {0}
      else:
        prefixes.add(prefix)

    return ans


# Link: https://leetcode.com/problems/beautiful-arrangement-ii/description/
class Solution:
  def constructArray(self, n: int, k: int) -> List[int]:
    ans = list(range(1, n - k + 1))

    for i in range(k):
      if i % 2 == 0:
        ans.append(n - i // 2)
      else:
        ans.append(n - k + (i + 1) // 2)

    return ans


# Link: https://leetcode.com/problems/find-longest-special-substring-that-occurs-thrice-ii/description/
class Solution:
  def maximumLength(self, s: str) -> int:
    n = len(s)
    runningLen = 0
    prevLetter = '@'
    # counts[i][j] := the frequency of ('a' + i) repeating j times
    counts = [[0] * (n + 1) for _ in range(26)]

    for c in s:
      if c == prevLetter:
        runningLen += 1
        counts[ord(c) - ord('a')][runningLen] += 1
      else:
        runningLen = 1
        counts[ord(c) - ord('a')][runningLen] += 1
        prevLetter = c

    def getMaxFreq(count: List[int]) -> int:
      """Returns the maximum frequency that occurs more than three times."""
      times = 0
      for freq in range(n, 0, -1):
        times += count[freq]
        if times >= 3:
          return freq
      return -1

    return max(getMaxFreq(count) for count in counts)


# Link: https://leetcode.com/problems/shortest-path-with-alternating-colors/description/
from enum import Enum


class Color(Enum):
  kInit = 0
  kRed = 1
  kBlue = 2


class Solution:
  def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
    ans = [-1] * n
    graph = [[] for _ in range(n)]  # graph[u] := [(v, edgeColor)]
    q = collections.deque([(0, Color.kInit)])  # [(u, prevColor)]

    for u, v in redEdges:
      graph[u].append((v, Color.kRed))

    for u, v in blueEdges:
      graph[u].append((v, Color.kBlue))

    step = 0
    while q:
      for _ in range(len(q)):
        u, prevColor = q.popleft()
        if ans[u] == -1:
          ans[u] = step
        for i, (v, edgeColor) in enumerate(graph[u]):
          if v == -1 or edgeColor == prevColor:
            continue
          q.append((v, edgeColor))
          graph[u][i] = (-1, edgeColor)  # Mark (u, v) as used.
      step += 1

    return ans


# Link: https://leetcode.com/problems/average-height-of-buildings-in-each-segment/description/
class Solution:
  def averageHeightOfBuildings(self, buildings: List[List[int]]) -> List[List[int]]:
    ans = []
    events = []

    for start, end, height in buildings:
      events.append((start, height))
      events.append((end, -height))

    prev = 0
    count = 0
    sumHeight = 0

    for curr, h in sorted(events):
      height = abs(h)
      if sumHeight > 0 and curr > prev:
        avgHeight = sumHeight // count
        if ans and ans[-1][1] == prev and avgHeight == ans[-1][2]:
          ans[-1][1] = curr
        else:
          ans.append([prev, curr, avgHeight])
      sumHeight += h
      count += 1 if h > 0 else -1
      prev = curr

    return ans


# Link: https://leetcode.com/problems/describe-the-painting/description/
from sortedcontainers import SortedDict


class Solution:
  def splitPainting(self, segments: List[List[int]]) -> List[List[int]]:
    ans = []
    prevIndex = 0
    runningMix = 0
    timeline = SortedDict()

    for start, end, color in segments:
      timeline[start] = timeline.get(start, 0) + color
      timeline[end] = timeline.get(end, 0) - color

    for i, mix in timeline.items():
      if runningMix > 0:
        ans.append([prevIndex, i, runningMix])
      runningMix += mix
      prevIndex = i

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-keypresses/description/
class Solution:
  def minimumKeypresses(self, s: str) -> int:
    return sum(c * (i // 9 + 1)
               for i, c in enumerate(sorted(Counter(s).values(), reverse=True)))


# Link: https://leetcode.com/problems/surrounded-regions/description/
class Solution:
  def solve(self, board: List[List[str]]) -> None:
    if not board:
      return

    m = len(board)
    n = len(board[0])

    def dfs(i: int, j: int) -> None:
      """Marks the grids with 'O' that stretch from the four sides to '*'."""
      if i < 0 or i == m or j < 0 or j == n:
        return
      if board[i][j] != 'O':
        return
      board[i][j] = '*'
      dfs(i + 1, j)
      dfs(i - 1, j)
      dfs(i, j + 1)
      dfs(i, j - 1)

    for i in range(m):
      for j in range(n):
        if i * j == 0 or i == m - 1 or j == n - 1:
          dfs(i, j)

    for row in board:
      for i, c in enumerate(row):
        row[i] = 'O' if c == '*' else 'X'


# Link: https://leetcode.com/problems/surrounded-regions/description/
class Solution:
  def solve(self, board: List[List[str]]) -> None:
    if not board:
      return

    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(board)
    n = len(board[0])
    q = collections.deque()

    for i in range(m):
      for j in range(n):
        if i * j == 0 or i == m - 1 or j == n - 1:
          if board[i][j] == 'O':
            q.append((i, j))
            board[i][j] = '*'

    # Mark the grids that stretch from the four sides with '*'.
    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if board[x][y] != 'O':
          continue
        q.append((x, y))
        board[x][y] = '*'

    for row in board:
      for i, c in enumerate(row):
        row[i] = 'O' if c == '*' else 'X'


# Link: https://leetcode.com/problems/largest-element-in-an-array-after-merge-operations/description/
class Solution:
  def maxArrayValue(self, nums: List[int]) -> int:
    ans = nums[-1]

    for i in range(len(nums) - 2, -1, -1):
      if nums[i] > ans:
        ans = nums[i]
      else:
        ans += nums[i]

    return ans


# Link: https://leetcode.com/problems/walking-robot-simulation/description/
class Solution:
  def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    ans = 0
    d = 0  # 0 := north, 1 := east, 2 := south, 3 := west
    x = 0  # the start x
    y = 0  # the start y
    obstaclesSet = {(x, y) for x, y in obstacles}

    for c in commands:
      if c == -1:
        d = (d + 1) % 4
      elif c == -2:
        d = (d + 3) % 4
      else:
        for _ in range(c):
          if (x + dirs[d], y + dirs[d + 1]) in obstaclesSet:
            break
          x += dirs[d]
          y += dirs[d + 1]

      ans = max(ans, x * x + y * y)

    return ans


# Link: https://leetcode.com/problems/minimum-swaps-to-arrange-a-binary-grid/description/
class Solution:
  def minSwaps(self, grid: List[List[int]]) -> int:
    n = len(grid)
    ans = 0
    # suffixZeros[i] := the number of suffix zeros in the i-th row
    suffixZeros = [n if 1 not in row else row[::-1].index(1) for row in grid]

    for i in range(n):
      neededZeros = n - 1 - i
      # Get the first row with suffix zeros >= `neededZeros` in suffixZeros[i:..n).
      j = next((j for j in range(i, n) if suffixZeros[j] >= neededZeros), -1)
      if j == -1:
        return -1
      # Move the rows[j] to the rows[i].
      for k in range(j, i, -1):
        suffixZeros[k] = suffixZeros[k - 1]
      ans += j - i

    return ans


# Link: https://leetcode.com/problems/find-root-of-n-ary-tree/description/
class Solution:
  def findRoot(self, tree: List['Node']) -> 'Node':
    sum = 0

    for node in tree:
      sum ^= node.val
      for child in node.children:
        sum ^= child.val

    for node in tree:
      if node.val == sum:
        return node


# Link: https://leetcode.com/problems/find-longest-special-substring-that-occurs-thrice-i/description/
class Solution:
  def maximumLength(self, s: str) -> int:
    n = len(s)
    runningLen = 0
    prevLetter = '@'
    # counts[i][j] := the frequency of ('a' + i) repeating j times
    counts = [[0] * (n + 1) for _ in range(26)]

    for c in s:
      if c == prevLetter:
        runningLen += 1
        counts[ord(c) - ord('a')][runningLen] += 1
      else:
        runningLen = 1
        counts[ord(c) - ord('a')][runningLen] += 1
        prevLetter = c

    def getMaxFreq(count: List[int]) -> int:
      """Returns the maximum frequency that occurs more than three times."""
      times = 0
      for freq in range(n, 0, -1):
        times += count[freq]
        if times >= 3:
          return freq
      return -1

    return max(getMaxFreq(count) for count in counts)


# Link: https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/description/
class Solution:
  def constructDistancedSequence(self, n: int) -> List[int]:
    ans = [0] * (2 * n - 1)

    def dfs(i: int, mask: int) -> bool:
      if i == len(ans):
        return True
      if ans[i] > 0:
        return dfs(i + 1, mask)

      # Greedily fill in `ans` in descending order.
      for num in range(n, 0, -1):
        if (mask >> num & 1) == 1:
          continue
        if num == 1:
          ans[i] = num
          if dfs(i + 1, mask | 1 << num):
            return True
          ans[i] = 0
        else:  # num in [2, n]
          if i + num >= len(ans) or ans[i + num] > 0:
            continue
          ans[i] = num
          ans[i + num] = num
          if dfs(i + 1, mask | 1 << num):
            return True
          ans[i + num] = 0
          ans[i] = 0

      return False

    dfs(0, 0)
    return ans


# Link: https://leetcode.com/problems/minimum-area-rectangle/description/
class Solution:
  def minAreaRect(self, points: List[List[int]]) -> int:
    ans = math.inf
    xToYs = collections.defaultdict(set)

    for x, y in points:
      xToYs[x].add(y)

    for i in range(len(points)):
      for j in range(i):
        x1, y1 = points[i]
        x2, y2 = points[j]
        if x1 == x2 or y1 == y2:
          continue
        if y2 in xToYs[x1] and y1 in xToYs[x2]:
          ans = min(ans, abs(x1 - x2) * abs(y1 - y2))

    return ans if ans < math.inf else 0


# Link: https://leetcode.com/problems/delete-leaves-with-a-given-value/description/
class Solution:
  def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
    if not root:
      return None
    root.left = self.removeLeafNodes(root.left, target)
    root.right = self.removeLeafNodes(root.right, target)
    return None if self._isLeaf(root) and root.val == target else root

  def _isLeaf(self, root: Optional[TreeNode]) -> bool:
    return not root.left and not root.right


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-all-array-elements-equal-to-1/description/
class Solution:
  def minOperations(self, nums: List[int]) -> int:
    n = len(nums)
    ones = nums.count(1)
    if ones > 0:
      return n - ones

    # the minimum operations to make the shortest subarray with a gcd == 1
    minOps = math.inf

    for i, g in enumerate(nums):
      for j in range(i + 1, n):
        g = math.gcd(g, nums[j])
        if g == 1:   # gcd(nums[i..j]:== 1
          minOps = min(minOps, j - i)
          break

    # After making the shortest subarray with `minOps`, need additional n - 1
    # operations to make the other numbers to 1.
    return -1 if minOps == math.inf else minOps + n - 1


# Link: https://leetcode.com/problems/watering-plants/description/
class Solution:
  def wateringPlants(self, plants: List[int], capacity: int) -> int:
    ans = 0
    currCapacity = 0

    for i, plant in enumerate(plants):
      if currCapacity + plant <= capacity:
        currCapacity += plant
      else:
        currCapacity = plant  # Reset
        ans += i * 2

    return ans + len(plants)


# Link: https://leetcode.com/problems/minimum-operations-to-make-array-equal/description/
class Solution:
  def minOperations(self, n: int) -> int:
    def arr(self, i: int) -> int:
      """Returns the i-th element of `arr`, where 1 <= i <= n."""
      return (i - 1) * 2 + 1

    #     median := median of arr
    #   diffs[i] := median - arr[i] where i <= i <= n // 2
    #        ans := sum(diffs)
    # e.g.
    # n = 5, arr = [1, 3, 5, 7, 9], diffs = [4, 2]
    #        ans = (4 + 2) * 2 // 2 = 6
    # n = 6, arr = [1, 3, 5, 7, 9, 11], diffs = [5, 3, 1]
    #        ans = (5 + 1) * 3 // 2 = 9
    halfSize = n // 2
    median = (arr(n) + arr(1)) // 2
    firstDiff = median - arr(1)
    lastDiff = median - arr(halfSize)
    return (firstDiff + lastDiff) * halfSize // 2


# Link: https://leetcode.com/problems/bitwise-and-of-numbers-range/description/
class Solution:
  def rangeBitwiseAnd(self, m: int, n: int) -> int:
    return self.rangeBitwiseAnd(m >> 1, n >> 1) << 1 if m < n else m


# Link: https://leetcode.com/problems/find-indices-with-index-and-value-difference-ii/description/
class Solution:
  def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
    # nums[minIndex] := the minimum number with enough index different from the current number
    minIndex = 0
    # nums[maxIndex] := the maximum number with enough index different from the current number
    maxIndex = 0

    for i in range(indexDifference, len(nums)):
      if nums[i - indexDifference] < nums[minIndex]:
        minIndex = i - indexDifference
      if nums[i - indexDifference] > nums[maxIndex]:
        maxIndex = i - indexDifference
      if nums[i] - nums[minIndex] >= valueDifference:
        return [i, minIndex]
      if nums[maxIndex] - nums[i] >= valueDifference:
        return [i, maxIndex]

    return [-1, -1]


# Link: https://leetcode.com/problems/advantage-shuffle/description/
from sortedcontainers import SortedList


class Solution:
  def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
    sl = SortedList(nums1)

    for i, num in enumerate(nums2):
      index = 0 if sl[-1] <= num else sl.bisect_right(num)
      nums1[i] = sl[index]
      del sl[index]

    return nums1


# Link: https://leetcode.com/problems/number-of-divisible-substrings/description/
class Solution:
  def countDivisibleSubstrings(self, word: str) -> int:
    # Let f(c) = d, where d = 1, 2, ..., 9.
    # Rephrase the question to return the number of substrings that satisfy
    #    f(c1) + f(c2) + ... + f(ck) // k = avg
    # => f(c1) + f(c2) + ... + f(ck) - k * avg, where avg in [1, 9].
    ans = 0

    def f(c: str) -> int:
      return 9 - (ord('z') - ord(c)) // 3

    for avg in range(1, 10):
      prefix = 0
      prefixCount = collections.Counter({0: 1})
      for c in word:
        prefix += f(c) - avg
        ans += prefixCount[prefix]
        prefixCount[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/find-duplicate-file-in-system/description/
class Solution:
  def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    contentToPathFiles = collections.defaultdict(list)

    for path in paths:
      words = path.split(' ')
      rootPath = words[0]  # "root/d1/d2/.../dm"
      for fileAndContent in words[1:]:  # "fn.txt(fn_content)"
        l = fileAndContent.find('(')
        r = fileAndContent.find(')')
        # "fn.txt"
        file = fileAndContent[:l]
        # "fn_content"
        content = fileAndContent[l + 1:r]
        # "root/d1/d2/.../dm/fn.txt"
        filePath = rootPath + '/' + file
        contentToPathFiles[content].append(filePath)

    return [filePath for filePath in contentToPathFiles.values() if len(filePath) > 1]


# Link: https://leetcode.com/problems/minimum-sideway-jumps/description/
class Solution:
  def minSideJumps(self, obstacles: List[int]) -> int:
    kInf = 1e6
    # dp[i] := the minimum jump to reach the i-th lane
    dp = [kInf, 1, 0, 1]

    for obstacle in obstacles:
      print(dp)
      if obstacle > 0:
        dp[obstacle] = kInf
      for i in range(1, 4):  # the current
        if i != obstacle:
          for j in range(1, 4):  # the previous
            dp[i] = min(dp[i], dp[j] + (0 if i == j else 1))

    return min(dp)


# Link: https://leetcode.com/problems/count-unhappy-friends/description/
class Solution:
  def unhappyFriends(self, n: int, preferences: List[List[int]], pairs: List[List[int]]) -> int:
    ans = 0
    matches = [0] * n
    prefer = [{} for _ in range(n)]

    for x, y in pairs:
      matches[x] = y
      matches[y] = x

    for i in range(n):
      for j in range(n - 1):
        prefer[i][preferences[i][j]] = j

    for x in range(n):
      for u in prefer[x].keys():
        y = matches[x]
        v = matches[u]
        if prefer[x][u] < prefer[x][y] and prefer[u][x] < prefer[u][v]:
          ans += 1
          break

    return ans


# Link: https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/description/
class Solution:
  def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
    minDistance = math.inf
    firstMaIndex = -1
    prevMaIndex = -1
    index = 1
    prev = head  # Point to the index 0.
    curr = head.next  # Point to the index 1.

    while curr.next:
      if curr.val > prev.val and curr.val > curr.next.val or \
         curr.val < prev.val and curr.val < curr.next.val:
        if firstMaIndex == -1:  # Only assign once.
          firstMaIndex = index
        if prevMaIndex != -1:
          minDistance = min(minDistance, index - prevMaIndex)
        prevMaIndex = index
      prev = curr
      curr = curr.next
      index += 1

    if minDistance == math.inf:
      return [-1, -1]
    return [minDistance, prevMaIndex - firstMaIndex]


# Link: https://leetcode.com/problems/minimum-operations-to-convert-number/description/
class Solution:
  def minimumOperations(self, nums: List[int], start: int, goal: int) -> int:
    ans = 0
    q = collections.deque([start])
    seen = {start}

    while q:
      ans += 1
      for _ in range(len(q)):
        x = q.popleft()
        for num in nums:
          for res in (x + num, x - num, x ^ num):
            if res == goal:
              return ans
            if res < 0 or res > 1000 or res in seen:
              continue
            seen.add(res)
            q.append(res)

    return -1


# Link: https://leetcode.com/problems/check-if-strings-can-be-made-equal-with-operations-ii/description/
class Solution:
  def checkStrings(self, s1: str, s2: str) -> bool:
    count = [collections.Counter() for _ in range(2)]

    for i, (a, b) in enumerate(zip(s1, s2)):
      count[i % 2][a] += 1
      count[i % 2][b] -= 1

    return all(freq == 0 for freq in count[0].values()) \
        and all(freq == 0 for freq in count[1].values())


# Link: https://leetcode.com/problems/implement-magic-dictionary/description/
class MagicDictionary:
  def __init__(self):
    self.dict = {}

  def buildDict(self, dictionary: List[str]) -> None:
    for word in dictionary:
      for i, c in enumerate(word):
        replaced = self._getReplaced(word, i)
        self.dict[replaced] = '*' if replaced in self.dict else c

  def search(self, searchWord: str) -> bool:
    for i, c in enumerate(searchWord):
      replaced = self._getReplaced(searchWord, i)
      if self.dict.get(replaced, c) != c:
        return True
    return False

  def _getReplaced(self, s: str, i: int) -> str:
    return s[:i] + '*' + s[i + 1:]


# Link: https://leetcode.com/problems/implement-magic-dictionary/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = collections.defaultdict(TrieNode)
    self.isWord = False


class MagicDictionary:
  def __init__(self):
    self.root = TrieNode()

  def buildDict(self, dictionary: List[str]) -> None:
    for word in dictionary:
      self._insert(word)

  def search(self, searchWord: str) -> bool:
    node: TrieNode = self.root
    for i, c in enumerate(searchWord):
      for letter in string.ascii_lowercase:
        if letter == c:
          continue
        child = node.children[letter]
        if not child:
          continue
        # Replace the searchWord[i] with `letter`, then check if
        # searchWord[i + 1..n) matches `child`.
        if self._find(child, searchWord, i + 1):
          return True
      if not node.children[c]:
        return False
      node = node.children[c]
    return False

  def _insert(self, word: str) -> None:
    node: TrieNode = self.root
    for c in word:
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True

  def _find(self, node: TrieNode, word: str, i: int) -> bool:
    for c in word[i:]:
      if c not in node.children:
        return False
      node = node.children[c]
    return node.isWord


# Link: https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/description/
class Solution:
  def longestPalindrome(self, words: List[str]) -> int:
    ans = 0
    count = [[0] * 26 for _ in range(26)]

    for a, b in words:
      i = ord(a) - ord('a')
      j = ord(b) - ord('a')
      if count[j][i]:
        ans += 4
        count[j][i] -= 1
      else:
        count[i][j] += 1

    for i in range(26):
      if count[i][i]:
        return ans + 2

    return ans


# Link: https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/description/
class Solution:
  def longestSubarray(self, nums: List[int]) -> int:
    ans = 0
    maxIndex = 0
    sameNumLength = 0

    for i, num in enumerate(nums):
      if nums[i] == nums[maxIndex]:
        sameNumLength += 1
        ans = max(ans, sameNumLength)
      elif nums[i] > nums[maxIndex]:
        maxIndex = i
        sameNumLength = 1
        ans = 1
      else:
        sameNumLength = 0

    return ans


# Link: https://leetcode.com/problems/maximum-sum-score-of-array/description/
class Solution:
  def maximumSumScore(self, nums: List[int]) -> int:
    ans = -math.inf
    prefix = 0
    summ = sum(nums)

    for num in nums:
      prefix += num
      ans = max(ans, prefix, summ - prefix + num)

    return ans


# Link: https://leetcode.com/problems/remove-stones-to-minimize-the-total/description/
class Solution:
  def minStoneSum(self, piles: List[int], k: int) -> int:
    maxHeap = [-pile for pile in piles]
    heapq.heapify(maxHeap)

    for _ in range(k):
      heapq.heapreplace(maxHeap, maxHeap[0] // 2)

    return -sum(maxHeap)


# Link: https://leetcode.com/problems/two-best-non-overlapping-events/description/
class Solution:
  def maxTwoEvents(self, events: List[List[int]]) -> int:
    ans = 0
    maxValue = 0
    evts = []  # (time, isStart, value)

    for s, e, v in events:
      evts.append((s, 1, v))
      evts.append((e + 1, 0, v))

    # When two events have the same time, the one is not start will be in the front
    evts.sort()

    for _, isStart, value in evts:
      if isStart:
        ans = max(ans, value + maxValue)
      else:
        maxValue = max(maxValue, value)

    return ans


# Link: https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/
class Solution:
  def shipWithinDays(self, weights: List[int], days: int) -> int:
    def canShip(shipCapacity: int) -> bool:
      shipDays = 1
      capacity = 0
      for weight in weights:
        if capacity + weight > shipCapacity:
          shipDays += 1
          capacity = weight
        else:
          capacity += weight
      return shipDays <= days

    l = max(weights)
    r = sum(weights)
    return bisect.bisect_left(range(l, r), True,
                              key=lambda m: canShip(m)) + l


# Link: https://leetcode.com/problems/rotated-digits/description/
class Solution:
  def rotatedDigits(self, n: int) -> int:
    def isGoodNumber(i: int) -> bool:
      isRotated = False

      for c in str(i):
        if c == '0' or c == '1' or c == '8':
          continue
        if c == '2' or c == '5' or c == '6' or c == '9':
          isRotated = True
        else:
          return False

      return isRotated

    return sum(isGoodNumber(i) for i in range(1, n + 1))


# Link: https://leetcode.com/problems/k-highest-ranked-items-within-a-price-range/description/
class Solution:
  def highestRankedKItems(self, grid: List[List[int]], pricing: List[int], start: List[int], k: int) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    low, high = pricing
    row, col = start
    ans = []

    if low <= grid[row][col] <= high:
      ans.append([row, col])
      if k == 1:
        return ans

    q = collections.deque([(row, col)])
    seen = {(row, col)}  # Mark as visited.

    while q:
      neighbors = []
      for _ in range(len(q)):
        i, j = q.popleft()
        for t in range(4):
          x = i + dirs[t]
          y = j + dirs[t + 1]
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if not grid[x][y] or (x, y) in seen:
            continue
          if low <= grid[x][y] <= high:
            neighbors.append([x, y])
          q.append((x, y))
          seen.add((x, y))
      neighbors.sort(key=lambda x: (grid[x[0]][x[1]], x[0], x[1]))
      for neighbor in neighbors:
        if len(ans) < k:
          ans.append(neighbor)
        if len(ans) == k:
          return ans

    return ans


# Link: https://leetcode.com/problems/redundant-connection/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> bool:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return False
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1
    return True

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
    uf = UnionFind(len(edges) + 1)

    for edge in edges:
      u, v = edge
      if not uf.unionByRank(u, v):
        return edge


# Link: https://leetcode.com/problems/group-shifted-strings/description/
class Solution:
  def groupStrings(self, strings: List[str]) -> List[List[str]]:
    keyToStrings = collections.defaultdict(list)

    def getKey(s: str) -> str:
      """
      Returns the key of 's' by pairwise calculation of differences.
      e.g. getKey("abc") -> "1,1" because diff(a, b) = 1 and diff(b, c) = 1.
      """
      diffs = []

      for i in range(1, len(s)):
        diff = (ord(s[i]) - ord(s[i - 1]) + 26) % 26
        diffs.append(str(diff))

      return ','.join(diffs)

    for s in strings:
      keyToStrings[getKey(s)].append(s)

    return keyToStrings.values()


# Link: https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/
class Solution:
  def countPalindromicSubsequence(self, s: str) -> int:
    ans = 0
    first = [len(s)] * 26
    last = [0] * 26

    for i, c in enumerate(s):
      index = ord(c) - ord('a')
      first[index] = min(first[index], i)
      last[index] = i

    for f, l in zip(first, last):
      if f < l:
        ans += len(set(s[f + 1:l]))

    return ans


# Link: https://leetcode.com/problems/minimum-time-to-repair-cars/description/
class Solution:
  def repairCars(self, ranks: List[int], cars: int) -> int:
    def numCarsFixed(minutes: int) -> int:
      #    r * n^2 = minutes
      # -> n = sqrt(minutes / r)
      return sum(int(sqrt(minutes // rank)) for rank in ranks)

    return bisect.bisect_left(
        range(0, min(ranks) * cars**2), cars,
        key=lambda m: numCarsFixed(m))


# Link: https://leetcode.com/problems/correct-a-binary-tree/description/
class Solution:
  def __init__(self):
    self.seen = set()

  def correctBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if root == None:
      return None
    if root.right and root.right.val in self.seen:
      return None
    self.seen.add(root.val)
    root.right = self.correctBinaryTree(root.right)
    root.left = self.correctBinaryTree(root.left)
    return root


# Link: https://leetcode.com/problems/minimum-score-by-changing-two-elements/description/
class Solution:
  def minimizeSum(self, nums: List[int]) -> int:
    nums.sort()
    # Can always change the number to any other number in `nums`, so `low` becomes 0.
    # Thus, rephrase the problem as finding the minimum `high`.
    highOfChangingTwoMins = nums[-1] - nums[2]
    highOfChangingTwoMaxs = nums[-3] - nums[0]
    highOfChangingMinAndMax = nums[-2] - nums[1]
    return min(highOfChangingTwoMins, highOfChangingTwoMaxs,
               highOfChangingMinAndMax)


# Link: https://leetcode.com/problems/height-of-special-binary-tree/description/
class Solution:
  def heightOfTree(self, root: Optional[TreeNode]) -> int:
    if not root:
      return 0
    # a leaf node
    if root.left and root.left.right == root:
      return 0
    return 1 + max(self.heightOfTree(root.left), self.heightOfTree(root.right))


# Link: https://leetcode.com/problems/compare-strings-by-frequency-of-the-smallest-character/description/
class Solution:
  def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
    ans = []
    wordsFreq = sorted([word.count(min(word)) for word in words])

    for q in queries:
      count = q.count(min(q))
      index = bisect.bisect(wordsFreq, count)
      ans.append(len(words) - index)

    return ans


# Link: https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/description/
class Solution:
  def pathInZigZagTree(self, label: int) -> List[int]:
    def boundarySum(level: int):
      return 2**level + 2**(level + 1) - 1

    ans = []

    for l in range(21):
      if 2**l > label:
        level = l - 1
        break

    if level & 1:
      label = boundarySum(level) - label

    for l in reversed(range(level + 1)):
      ans.append(boundarySum(l) - label if l & 1 else label)
      label //= 2

    return ans[::-1]


# Link: https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
class Solution:
  def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    minHeap = []  # (matrix[i][j], i, j)

    i = 0
    while i < k and i < len(matrix):
      heapq.heappush(minHeap, (matrix[i][0], i, 0))
      i += 1

    while k > 1:
      k -= 1
      _, i, j = heapq.heappop(minHeap)
      if j + 1 < len(matrix[0]):
        heapq.heappush(minHeap, (matrix[i][j + 1], i, j + 1))

    return minHeap[0][0]


# Link: https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
class Solution:
  def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    def numsNoGreaterThan(m: int) -> int:
      count = 0
      j = len(matrix[0]) - 1
      # For each row, find the first index j s.t. row[j] <= m s.t. the number of
      # numbers <= m for this row will be j + 1.
      for row in matrix:
        while j >= 0 and row[j] > m:
          j -= 1
        count += j + 1
      return count

    return bisect.bisect_left(range(matrix[0][0], matrix[-1][-1]), k,
                              key=lambda m: numsNoGreaterThan(m)) + matrix[0][0]


# Link: https://leetcode.com/problems/reorder-data-in-log-files/description/
class Solution:
  def reorderLogFiles(self, logs: List[str]) -> List[str]:
    digitLogs = []
    letterLogs = []

    for log in logs:
      i = log.index(' ')
      if log[i + 1].isdigit():
        digitLogs.append(log)
      else:
        letterLogs.append((log[:i], log[i + 1:]))

    letterLogs.sort(key=lambda l: (l[1], l[0]))

    return [identifier + ' ' + letters for identifier, letters in letterLogs] + digitLogs


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/description/
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def getLCA(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
      if not root or root == p or root == q:
        return root
      left = getLCA(root.left, p, q)
      right = getLCA(root.right, p, q)
      if left and right:
        return root
      return left or right

    ans = getLCA(root, p, q)
    if ans == p:  # Search q in the subtree rooted at p.
      return ans if getLCA(p, q, q) else None
    if ans == q:  # Search p in the subtree rooted at q.
      return ans if getLCA(q, p, p) else None
    return ans  # (ans != p and ans != q) or ans is None


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/description/
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    seenP = False
    seenQ = False

    def getLCA(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
      nonlocal seenP
      nonlocal seenQ
      if not root:
        return None
      # Need to traverse the entire tree to update `seenP` and `seenQ`.
      left = getLCA(root.left, p, q)
      right = getLCA(root.right, p, q)
      if root == p:
        seenP = True
        return root
      if root == q:
        seenQ = True
        return root
      if left and right:
        return root
      return left or right

    lca = getLCA(root, p, q)
    return lca if seenP and seenQ else None


# Link: https://leetcode.com/problems/koko-eating-bananas/description/
class Solution:
  def minEatingSpeed(self, piles: List[int], h: int) -> int:
    # Returns true if Koko can eat all piles with speed m.
    def eatHours(m: int) -> bool:
      return sum((pile - 1) // m + 1 for pile in piles) <= h
    return bisect.bisect_left(range(1, max(piles)), True,
                              key=lambda m: eatHours(m)) + 1


# Link: https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/description/
class Solution:
  def getSmallestString(self, n: int, k: int) -> str:
    ans = []

    for i in range(n):
      remainingLetters = n - 1 - i
      rank = max(1, k - remainingLetters * 26)
      ans.append(chr(ord('a') + rank - 1))
      k -= rank

    return ''.join(ans)


# Link: https://leetcode.com/problems/substring-xor-queries/description/
class Solution:
  def substringXorQueries(self, s: str, queries: List[List[int]]) -> List[List[int]]:
    kMaxBit = 30
    # {val: [left, right]} := s[left..right]'s decimal value = val
    valToLeftAndRight = collections.defaultdict(lambda: [-1, -1])

    for left, c in enumerate(s):
      val = 0
      if c == '0':
        # edge case: Save the index of the first 0.
        if 0 not in valToLeftAndRight:
          valToLeftAndRight[0] = [left, left]
        continue
      for right in range(left, min(len(s), left + kMaxBit)):
        val = val * 2 + int(s[right])
        if val not in valToLeftAndRight:
          valToLeftAndRight[val] = [left, right]

    return [valToLeftAndRight[first, right]
            for first, right in queries]


# Link: https://leetcode.com/problems/split-linked-list-in-parts/description/
class Solution:
  def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
    ans = [[] for _ in range(k)]
    length = 0
    curr = root
    while curr:
      length += 1
      curr = curr.next
    subLength = length // k
    remainder = length % k

    prev = None
    head = root

    for i in range(k):
      ans[i] = head
      for j in range(subLength + (1 if remainder > 0 else 0)):
        prev = head
        head = head.next
      if prev:
        prev.next = None
      remainder -= 1

    return ans


# Link: https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/
class Solution:
  def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
      return []

    ans = []
    dq = collections.deque([root])
    isLeftToRight = True

    while dq:
      currLevel = []
      for _ in range(len(dq)):
        if isLeftToRight:
          node = dq.popleft()
          currLevel.append(node.val)
          if node.left:
            dq.append(node.left)
          if node.right:
            dq.append(node.right)
        else:
          node = dq.pop()
          currLevel.append(node.val)
          if node.right:
            dq.appendleft(node.right)
          if node.left:
            dq.appendleft(node.left)
      ans.append(currLevel)
      isLeftToRight = not isLeftToRight

    return ans


# Link: https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/
class Solution:
  def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
      return []

    ans = []
    q = collections.deque([root])
    isLeftToRight = True

    while q:
      size = len(q)
      currLevel = [0] * size
      for i in range(size):
        node = q.popleft()
        index = i if isLeftToRight else size - i - 1
        currLevel[index] = node.val
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      ans.append(currLevel)
      isLeftToRight = not isLeftToRight

    return ans


# Link: https://leetcode.com/problems/number-of-nodes-with-value-one/description/
class Solution:
  def numberOfNodes(self, n: int, queries: List[int]) -> int:
    # flipped[i] := True if we should flip all the values in the subtree rooted
    # at i
    flipped = [False] * (n + 1)

    for query in queries:
      flipped[query] = flipped[query] ^ True

    def dfs(label: int, value: int) -> int:
      if label > n:
        return 0
      value ^= flipped[label]
      return value + dfs(label * 2, value) + dfs(label * 2 + 1, value)

    return dfs(1, 0)


# Link: https://leetcode.com/problems/distant-barcodes/description/
class Solution:
  def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
    ans = [0] * len(barcodes)
    count = collections.Counter(barcodes)
    i = 0  # ans' index
    maxNum = max(count, key=count.get)

    def fillAns(num: int) -> None:
      nonlocal i
      while count[num]:
        ans[i] = num
        i = i + 2 if i + 2 < len(barcodes) else 1
        count[num] -= 1

    fillAns(maxNum)
    for num in count.keys():
      fillAns(num)

    return ans


# Link: https://leetcode.com/problems/knight-probability-in-chessboard/description/
class Solution:
  def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
    dirs = ((1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    # dp[i][j] := the probability to stand on (i, j)
    dp = [[0] * n for _ in range(n)]
    dp[row][column] = 1.0

    for _ in range(k):
      newDp = [[0] * n for _ in range(n)]
      for i in range(n):
        for j in range(n):
          for dx, dy in dirs:
            x = i + dx
            y = j + dy
            if 0 <= x < n and 0 <= y < n:
              newDp[i][j] += dp[x][y]
      dp = newDp

    return sum(map(sum, dp)) / 8**k


# Link: https://leetcode.com/problems/validate-binary-search-tree/description/
class Solution:
  def isValidBST(self, root: Optional[TreeNode]) -> bool:
    def isValidBST(root: Optional[TreeNode],
                   minNode: Optional[TreeNode], maxNode: Optional[TreeNode]) -> bool:
      if not root:
        return True
      if minNode and root.val <= minNode.val:
        return False
      if maxNode and root.val >= maxNode.val:
        return False

      return isValidBST(root.left, minNode, root) and \
          isValidBST(root.right, root, maxNode)

    return isValidBST(root, None, None)


# Link: https://leetcode.com/problems/validate-binary-search-tree/description/
class Solution:
  def isValidBST(self, root: Optional[TreeNode]) -> bool:
    stack = []
    pred = None

    while root or stack:
      while root:
        stack.append(root)
        root = root.left
      root = stack.pop()
      if pred and pred.val >= root.val:
        return False
      pred = root
      root = root.right

    return True


# Link: https://leetcode.com/problems/minimum-size-subarray-in-infinite-array/description/
class Solution:
  def minSizeSubarray(self, nums: List[int], target: int) -> int:
    summ = sum(nums)
    n = len(nums)
    remainingTarget = target % summ
    repeatLength = (target // summ) * n
    if remainingTarget == 0:
      return repeatLength

    suffixPlusPrefixLength = n
    prefix = 0
    prefixToIndex = {0: -1}

    for i in range(2 * n):
      prefix += nums[i % n]
      if prefix - remainingTarget in prefixToIndex:
        suffixPlusPrefixLength = min(
            suffixPlusPrefixLength,
            i - prefixToIndex[prefix - remainingTarget])
      prefixToIndex[prefix] = i

    return -1 if suffixPlusPrefixLength == n else suffixPlusPrefixLength + repeatLength


# Link: https://leetcode.com/problems/fruit-into-baskets/description/
class Solution:
  def totalFruit(self, fruits: List[int]) -> int:
    ans = 0
    count = collections.defaultdict(int)

    l = 0
    for r, fruit in enumerate(fruits):
      count[fruit] += 1
      while len(count) > 2:
        count[fruits[l]] -= 1
        if count[fruits[l]] == 0:
          del count[fruits[l]]
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/partition-labels/description/
class Solution:
  def partitionLabels(self, s: str) -> List[int]:
    ans = []
    letterToRightmostIndex = {c: i for i, c in enumerate(s)}

    l = 0  # the leftmost index of the current running string
    r = 0  # the rightmost index of the current running string

    for i, c in enumerate(s):
      r = max(r, letterToRightmostIndex[c])
      if i == r:
        ans.append(r - l + 1)
        l = r + 1

    return ans


# Link: https://leetcode.com/problems/maximum-font-to-fit-a-sentence-in-a-screen/description/
# """
# This is FontInfo's API interface.
# You should not implement it, or speculate about its implementation
# """
# class FontInfo(object):
#   Return the width of char ch when fontSize is used.
#   def getWidth(self, fontSize: int, ch: str) -> int:
#     pass
#
#   def getHeight(self, fontSize: int) -> int:
#     pass
class Solution:
  def maxFont(self, text: str, w: int, h: int, fonts: List[int], fontInfo: 'FontInfo') -> int:
    count = collections.Counter(text)
    l = 0
    r = len(fonts) - 1

    while l < r:
      m = (l + r + 1) // 2
      if fontInfo.getHeight(fonts[m]) <= h and self._getWidthSum(count, fonts[m], fontInfo) <= w:
        l = m
      else:
        r = m - 1

    return fonts[l] if self._getWidthSum(count, fonts[l], fontInfo) <= w else -1

  def _getWidthSum(self, count: List[int], font: int, fontInfo: 'FontInfo') -> int:
    width = 0
    for c in string.ascii_lowercase:
      width += count[c] * fontInfo.getWidth(font, c)
    return width


# Link: https://leetcode.com/problems/range-product-queries-of-powers/description/
class Solution:
  def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
    kMod = 1_000_000_007
    kMaxBit = 30
    ans = []
    powers = [1 << i for i in range(kMaxBit) if n >> i & 1]

    for left, right in queries:
      prod = 1
      for i in range(left, right + 1):
        prod *= powers[i]
        prod %= kMod
      ans.append(prod)

    return ans


# Link: https://leetcode.com/problems/sum-in-a-matrix/description/
class Solution:
  def matrixSum(self, nums: List[List[int]]) -> int:
    for row in nums:
      row.sort()

    return sum(max(col) for col in zip(*nums))


# Link: https://leetcode.com/problems/max-area-of-island/description/
class Solution:
  def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    def dfs(i: int, j: int) -> int:
      if i < 0 or i == len(grid) or j < 0 or j == len(grid[0]):
        return 0
      if grid[i][j] != 1:
        return 0

      grid[i][j] = 2

      return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

    return max(dfs(i, j) for i in range(len(grid)) for j in range(len(grid[0])))


# Link: https://leetcode.com/problems/maximum-swap/description/
class Solution:
  def maximumSwap(self, num: int) -> int:
    s = list(str(num))
    dict = {c: i for i, c in enumerate(s)}

    for i, c in enumerate(s):
      for digit in reversed(string.digits):
        if digit <= c:
          break
        if digit in dict and dict[digit] > i:
          s[i], s[dict[digit]] = digit, s[i]
          return int(''.join(s))

    return num


# Link: https://leetcode.com/problems/find-the-length-of-the-longest-common-prefix/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}


class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, word: str) -> None:
    node: TrieNode = self.root
    for c in word:
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True

  def search(self, word: str) -> int:
    prefixLength = 0
    node = self.root
    for c in word:
      if c not in node.children:
        break
      node = node.children[c]
      prefixLength += 1
    return prefixLength


class Solution:
  def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
    trie = Trie()

    for num in arr1:
      trie.insert(str(num))

    return max(trie.search(str(num)) for num in arr2)


# Link: https://leetcode.com/problems/kth-smallest-subarray-sum/description/
class Solution:
  def kthSmallestSubarraySum(self, nums: List[int], k: int) -> int:
    def numSubarrayLessThan(m: int) -> int:
      res = 0
      summ = 0
      l = 0
      for r, num in enumerate(nums):
        summ += num
        while summ > m:
          summ -= nums[l]
          l += 1
        res += r - l + 1
      return res

    return bisect.bisect_left(range(0, sum(nums)), k,
                              key=lambda m: numSubarrayLessThan(m))


# Link: https://leetcode.com/problems/reduction-operations-to-make-the-array-elements-equal/description/
class Solution:
  def reductionOperations(self, nums: List[int]) -> int:
    ans = 0

    nums.sort()

    for i in range(len(nums) - 1, 0, -1):
      if nums[i] != nums[i - 1]:
        ans += len(nums) - i

    return ans


# Link: https://leetcode.com/problems/mice-and-cheese/description/
class Solution:
  def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
    return sum(reward2) + sum(heapq.nlargest(k, (a - b for a, b in zip(reward1, reward2))))


# Link: https://leetcode.com/problems/palindrome-permutation-ii/description/
class Solution:
  def generatePalindromes(self, s: str) -> List[str]:
    count = collections.Counter(s)

    # Count odd ones.
    odd = sum(value & 1 for value in count.values())

    # Can't form any palindrome.
    if odd > 1:
      return []

    ans = []
    candidates = []
    mid = ''

    # Get the mid and the candidates characters.
    for key, value in count.items():
      if value & 1:
        mid += key
      for _ in range(value // 2):
        candidates.append(key)

    def dfs(used: List[bool], path: List[chr]) -> None:
      """Generates all the unique palindromes from the candidates."""
      if len(path) == len(candidates):
        ans.append(''.join(path) + mid + ''.join(reversed(path)))
        return

      for i, candidate in enumerate(candidates):
        if used[i]:
          continue
        if i > 0 and candidate == candidates[i - 1] and not used[i - 1]:
          continue
        used[i] = True
        path.append(candidate)
        dfs(used, path)
        path.pop()
        used[i] = False

    # Backtrack to generate the ans strings.
    dfs([False] * len(candidates), [])
    return ans


# Link: https://leetcode.com/problems/campus-bikes/description/
class Solution:
  def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
    ans = [-1] * len(workers)
    usedBikes = [False] * len(bikes)
    # buckets[k] := (i, j), where k = dist(workers[i], bikes[j])
    buckets = [[] for _ in range(2001)]

    def dist(p1: List[int], p2: List[int]) -> int:
      return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    for i, worker in enumerate(workers):
      for j, bike in enumerate(bikes):
        buckets[dist(worker, bike)].append((i, j))

    for k in range(2001):
      for i, j in buckets[k]:
        if ans[i] == -1 and not usedBikes[j]:
          ans[i] = j
          usedBikes[j] = True

    return ans


# Link: https://leetcode.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers/description/
class Solution:
  def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
    def countTriplets(A: List[int], B: List[int]):
      """Returns the number of triplet (i, j, k) if A[i]^2 == B[j] * B[k]."""
      res = 0
      count = collections.Counter(B)

      for a in A:
        target = a * a
        for b, freq in count.items():
          if target % b > 0 or target // b not in count:
            continue
          if target // b == b:
            res += freq * (freq - 1)
          else:
            res += freq * count[target // b]

      return res // 2

    return countTriplets(nums1, nums2) + countTriplets(nums2, nums1)


# Link: https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/description/
class Solution:
  def longestSubstring(self, s: str, k: int) -> int:
    def longestSubstringWithNUniqueLetters(n: int) -> int:
      res = 0
      uniqueLetters = 0  # the number of unique letters
      lettersHavingKFreq = 0  # the number of letters having frequency >= k
      count = collections.Counter()

      l = 0
      for r, c in enumerate(s):
        if count[c] == 0:
          uniqueLetters += 1
        count[c] += 1
        if count[c] == k:
          lettersHavingKFreq += 1
        while uniqueLetters > n:
          if count[s[l]] == k:
            lettersHavingKFreq -= 1
          count[s[l]] -= 1
          if count[s[l]] == 0:
            uniqueLetters -= 1
          l += 1
        # Since both the number of unique letters and the number of letters
        # having frequency >= k are equal to n, this is a valid window.
        if lettersHavingKFreq == n:  # Implicit: uniqueLetters == n
          res = max(res, r - l + 1)

      return res

    return max(longestSubstringWithNUniqueLetters(n)
               for n in range(1, 27))


# Link: https://leetcode.com/problems/count-zero-request-servers/description/
class IndexedQuery:
  def __init__(self, queryIndex: int, query: int):
    self.queryIndex = queryIndex
    self.query = query

  def __iter__(self):
    yield self.queryIndex
    yield self.query


class Solution:
  def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
    ans = [0] * len(queries)
    count = [0] * (n + 1)

    logs.sort(key=lambda log: log[1])

    i = 0
    j = 0
    servers = 0

    # For each query, we care about logs[i..j].
    for queryIndex, query in sorted([IndexedQuery(i, query)
                                     for i, query in enumerate(queries)],
                                    key=lambda iq: iq.query):
      while j < len(logs) and logs[j][1] <= query:
        count[logs[j][0]] += 1
        if count[logs[j][0]] == 1:
          servers += 1
        j += 1
      while i < len(logs) and logs[i][1] < query - x:
        count[logs[i][0]] -= 1
        if count[logs[i][0]] == 0:
          servers -= 1
        i += 1
      ans[queryIndex] = n - servers

    return ans


# Link: https://leetcode.com/problems/longest-ideal-subsequence/description/
class Solution:
  def longestIdealString(self, s: str, k: int) -> int:
    # dp[i] := the longest subsequence that ends in ('a' + i)
    dp = [0] * 26

    for c in s:
      i = ord(c) - ord('a')
      dp[i] = 1 + self._getMaxReachable(dp, i, k)

    return max(dp)

  def _getMaxReachable(self, dp: List[int], i: int, k: int) -> int:
    first = max(0, i - k)
    last = min(25, i + k)
    maxReachable = 0
    for j in range(first, last + 1):
      maxReachable = max(maxReachable, dp[j])
    return maxReachable


# Link: https://leetcode.com/problems/find-unique-binary-string/description/
class Solution:
  def findDifferentBinaryString(self, nums: List[str]) -> str:
    return ''.join('1' if num[i] == '0' else '0' for i, num in enumerate(nums))


# Link: https://leetcode.com/problems/find-unique-binary-string/description/
class Solution:
  def findDifferentBinaryString(self, nums: List[str]) -> str:
    bitSize = len(nums[0])
    maxNum = 1 << bitSize
    numsSet = {int(num, 2) for num in nums}

    for num in range(maxNum):
      if num not in numsSet:
        return f'{num:0>{bitSize}b}'


# Link: https://leetcode.com/problems/sum-of-subarray-ranges/description/
class Solution:
  def subArrayRanges(self, nums: List[int]) -> int:
    n = len(nums)

    def sumSubarray(A: List[int], op):
      ans = 0
      prev = [-1] * n
      next = [n] * n
      stack = []

      for i, a in enumerate(A):
        while stack and op(A[stack[-1]], a):
          index = stack.pop()
          next[index] = i
        if stack:
          prev[i] = stack[-1]
        stack.append(i)

      for i, a in enumerate(A):
        ans += a * (i - prev[i]) * (next[i] - i)

      return ans

    return sumSubarray(nums, operator.lt) - sumSubarray(nums, operator.gt)


# Link: https://leetcode.com/problems/largest-number/description/
class LargerStrKey(str):
  def __lt__(x: str, y: str) -> bool:
    return x + y > y + x


class Solution:
  def largestNumber(self, nums: List[int]) -> str:
    return ''.join(sorted(map(str, nums), key=LargerStrKey)).lstrip('0') or '0'


# Link: https://leetcode.com/problems/longest-uncommon-subsequence-ii/description/
class Solution:
  def findLUSlength(self, strs: List[str]) -> int:
    def isSubsequence(a: str, b: str) -> bool:
      i = 0
      j = 0

      while i < len(a) and j < len(b):
        if a[i] == b[j]:
          i += 1
        j += 1

      return i == len(a)

    seen = set()
    duplicates = set()

    for s in strs:
      if s in seen:
        duplicates.add(s)
      seen.add(s)

    strs.sort(key=lambda s: -len(s))

    for i in range(len(strs)):
      if strs[i] in duplicates:
        continue
      isASubsequence = False
      for j in range(i):
        isASubsequence |= isSubsequence(strs[i], strs[j])
      if not isASubsequence:
        return len(strs[i])

    return -1


# Link: https://leetcode.com/problems/maximum-length-of-semi-decreasing-subarrays/description/
class Solution:
  def maxSubarrayLength(self, nums: List[int]) -> int:
    ans = 0
    stack = []

    for i in range(len(nums) - 1, -1, -1):
      # If nums[stack[-1]] <= nums[i], stack[-1] is better than i.
      # So, no need to append it.
      if not stack or nums[stack[-1]] > nums[i]:
        stack.append(i)

    for i, num in enumerate(nums):
      while stack and num > nums[stack[-1]]:
        ans = max(ans, stack.pop() - i + 1)

    return ans


# Link: https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description/
class Solution:
  def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
    ans = []

    nums.sort()

    for i in range(2, len(nums), 3):
      if nums[i] - nums[i - 2] > k:
        return []
      ans.append([nums[i - 2], nums[i - 1], nums[i]])

    return ans


# Link: https://leetcode.com/problems/maximum-value-after-insertion/description/
class Solution:
  def maxValue(self, n: str, x: int) -> str:
    isNegative = n[0] == '-'

    for i, c in enumerate(n):
      if not isNegative and ord(c) - ord('0') < x or isNegative and ord(c) - ord('0') > x:
        return n[:i] + str(x) + n[i:]

    return n + str(x)


# Link: https://leetcode.com/problems/check-if-move-is-legal/description/
class Solution:
  def checkMove(self, board: List[List[str]], rMove: int, cMove: int, color: str) -> bool:
    dirs = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1))

    for dx, dy in dirs:
      cellsCount = 2
      i = rMove + dx
      j = cMove + dy
      while 0 <= i < 8 and 0 <= j < 8:
        # There are no free cells in between.
        if board[i][j] == '.':
          break
        # Need >= 3 cells.
        if cellsCount == 2 and board[i][j] == color:
          break
        # >= 3 cells.
        if board[i][j] == color:
          return True
        i += dx
        j += dy
        cellsCount += 1

    return False


# Link: https://leetcode.com/problems/find-leaves-of-binary-tree/description/
class Solution:
  def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
    ans = []

    def depth(root: Optional[TreeNode]) -> int:
      """Returns the depth of the root (0-indexed)."""
      if not root:
        return -1

      l = depth(root.left)
      r = depth(root.right)
      h = 1 + max(l, r)

      if len(ans) == h:  # Meet a leaf
        ans.append([])

      ans[h].append(root.val)
      return h

    depth(root)
    return ans


# Link: https://leetcode.com/problems/keys-and-rooms/description/
class Solution:
  def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    seen = [False] * len(rooms)

    def dfs(node: int) -> None:
      seen[node] = True
      for child in rooms[node]:
        if not seen[child]:
          dfs(child)

    dfs(0)
    return all(seen)


# Link: https://leetcode.com/problems/task-scheduler/description/
class Solution:
  def leastInterval(self, tasks: List[str], n: int) -> int:
    count = collections.Counter(tasks)
    maxFreq = max(count.values())
    # Put the most frequent task in the slot first.
    maxFreqTaskOccupy = (maxFreq - 1) * (n + 1)
    # Get the number of tasks with same frequency as maxFreq, we'll append them after the
    # `maxFreqTaskOccupy`.
    nMaxFreq = sum(value == maxFreq for value in count.values())
    # max(
    #   the most frequent task is frequent enough to force some idle slots,
    #   the most frequent task is not frequent enough to force idle slots
    # )
    return max(maxFreqTaskOccupy + nMaxFreq, len(tasks))


# Link: https://leetcode.com/problems/number-of-ways-to-build-sturdy-brick-wall/description/
class Solution:
  def buildWall(self, height: int, width: int, bricks: List[int]) -> int:
    kMod = 1_000_000_007
    # Stores the valid rows in bitmask.
    rows = []
    self._buildRows(width, bricks, 0, rows)

    n = len(rows)
    # dp[i] := the number of ways to build `h` height walls with rows[i] in the bottom
    dp = [1] * n
    # graph[i] := the valid neighbors of rows[i]
    graph = [[] for _ in range(n)]

    for i, a in enumerate(rows):
      for j, b in enumerate(rows):
        if not a & b:
          graph[i].append(j)

    for _ in range(2, height + 1):
      newDp = [0] * n
      for i in range(n):
        for v in graph[i]:
          newDp[i] += dp[v]
          newDp[i] %= kMod
      dp = newDp

    return sum(dp) % kMod

  def _buildRows(self, width: int, bricks: List[int], path: int, rows: List[int]):
    for brick in bricks:
      if brick == width:
        rows.append(path)
      elif brick < width:
        newWidth = width - brick
        self._buildRows(newWidth, bricks, path | 2 << newWidth, rows)


# Link: https://leetcode.com/problems/count-words-obtained-after-adding-a-letter/description/
class Solution:
  def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
    def getMask(s: str) -> int:
      mask = 0
      for c in s:
        mask ^= 1 << ord(c) - ord('a')
      return mask

    ans = 0
    seen = set(getMask(w) for w in startWords)

    for targetWord in targetWords:
      mask = getMask(targetWord)
      for c in targetWord:
        # Toggle one character.
        if mask ^ 1 << ord(c) - ord('a') in seen:
          ans += 1
          break

    return ans


# Link: https://leetcode.com/problems/decoded-string-at-index/description/
class Solution:
  def decodeAtIndex(self, s: str, k: int) -> str:
    size = 0

    for c in s:
      if c.isdigit():
        size *= int(c)
      else:
        size += 1

    for c in reversed(s):
      k %= size
      if k == 0 and c.isalpha():
        return c
      if c.isdigit():
        size //= int(c)
      else:
        size -= 1


# Link: https://leetcode.com/problems/expressive-words/description/
class Solution:
  def expressiveWords(self, s: str, words: List[str]) -> int:
    def isStretchy(word: str) -> bool:
      n = len(s)
      m = len(word)

      j = 0
      for i in range(n):
        if j < m and s[i] == word[j]:
          j += 1
        elif i > 1 and s[i] == s[i - 1] == s[i - 2]:
          continue
        elif 0 < i < n - 1 and s[i - 1] == s[i] == s[i + 1]:
          continue
        else:
          return False

      return j == m

    return sum(isStretchy(word) for word in words)


# Link: https://leetcode.com/problems/subarray-product-less-than-k/description/
class Solution:
  def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    if k <= 1:
      return 0

    ans = 0
    prod = 1

    j = 0
    for i, num in enumerate(nums):
      prod *= num
      while prod >= k:
        prod /= nums[j]
        j += 1
      ans += i - j + 1

    return ans


# Link: https://leetcode.com/problems/count-nice-pairs-in-an-array/description/
class Solution:
  def countNicePairs(self, nums: List[int]) -> int:
    freqs = collections.Counter(num - int(str(num)[::-1]) for num in nums)
    return sum(freq * (freq - 1) // 2 for freq in freqs.values()) % 1000000007


# Link: https://leetcode.com/problems/minimum-path-cost-in-a-grid/description/
class Solution:
  def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    # dp[i][j] := the minimum cost to reach grid[i][j]
    dp = [[math.inf] * n for _ in range(m)]
    dp[0] = grid[0]

    for i in range(1, m):
      for j in range(n):
        for k in range(n):
          dp[i][j] = min(dp[i][j], dp[i - 1][k] +
                         moveCost[grid[i - 1][k]][j] + grid[i][j])

    return min(dp[-1])


# Link: https://leetcode.com/problems/lexicographical-numbers/description/
class Solution:
  def lexicalOrder(self, n: int) -> List[int]:
    ans = []
    curr = 1

    while len(ans) < n:
      ans.append(curr)
      if curr * 10 <= n:
        curr *= 10
      else:
        while curr % 10 == 9 or curr == n:
          curr //= 10
        curr += 1

    return ans


# Link: https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips-ii/description/
class Solution:
  def removeOnes(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    maxMask = 1 << m * n
    # dp[i] := the minimum number of operations to remove all 1s from the grid,
    # where `i` is the bitmask of the state of the grid
    dp = [math.inf] * maxMask
    dp[0] = 0

    for mask in range(maxMask):
      for i in range(m):
        for j in range(n):
          if grid[i][j] == 1:
            nextMask = mask
            # Set the cells in the same row with 0.
            for k in range(n):
              nextMask &= ~(1 << i * n + k)
            # Set the cells in the same column with 0.
            for k in range(m):
              nextMask &= ~(1 << k * n + j)
            dp[mask] = min(dp[mask], 1 + dp[nextMask])

    return dp[self.encode(grid, m, n)]

  def encode(self, grid: List[List[int]], m: int, n: int) -> int:
    encoded = 0
    for i in range(m):
      for j in range(n):
        if grid[i][j]:
          encoded |= 1 << i * n + j
    return encoded


# Link: https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips-ii/description/
class Solution:
  def removeOnes(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    @functools.lru_cache(None)
    def dp(mask: int) -> int:
      """
      Returns the minimum number of operations to remove all 1s from the grid,
      where `mask` is the bitmask of the state of the grid.
      """
      if mask == 0:
        return 0
      ans = math.inf
      for i in range(m):
        for j in range(n):
          if mask >> i * n + j & 1:  # grid[i][j] == 1
            nextMask = mask
            for k in range(n):  # Set the cells in the same row with 0.
              nextMask &= ~(1 << i * n + k)
            for k in range(m):  # Set the cells in the same column with 0.
              nextMask &= ~(1 << k * n + j)
            ans = min(ans, 1 + dp(nextMask))
      return ans

    return dp(self.encode(grid, m, n))

  def encode(self, grid: List[List[int]], m: int, n: int) -> int:
    encoded = 0
    for i in range(m):
      for j in range(n):
        if grid[i][j]:
          encoded |= 1 << i * n + j
    return encoded


# Link: https://leetcode.com/problems/employee-importance/description/
class Solution:
  def getImportance(self, employees: List['Employee'], id: int) -> int:
    idToEmployee = {employee.id: employee for employee in employees}

    def dfs(id: int) -> int:
      values = idToEmployee[id].importance
      for subId in idToEmployee[id].subordinates:
        values += dfs(subId)
      return values

    return dfs(id)


# Link: https://leetcode.com/problems/minimum-moves-to-make-array-complementary/description/
class Solution:
  def minMoves(self, nums: List[int], limit: int) -> int:
    n = len(nums)
    ans = n
    # delta[i] := the number of moves needed when target goes from i - 1 to i
    delta = [0] * (limit * 2 + 2)

    for i in range(n // 2):
      a = nums[i]
      b = nums[n - 1 - i]
      delta[min(a, b) + 1] -= 1
      delta[a + b] -= 1
      delta[a + b + 1] += 1
      delta[max(a, b) + limit + 1] += 1

    # Initially, we need `moves` when the target is 2.
    moves = n
    for i in range(2, limit * 2 + 1):
      moves += delta[i]
      ans = min(ans, moves)

    return ans


# Link: https://leetcode.com/problems/elimination-game/description/
class Solution:
  def lastRemaining(self, n: int) -> int:
    return 1 if n == 1 else 2 * (1 + n // 2 - self.lastRemaining(n // 2))


# Link: https://leetcode.com/problems/binary-tree-right-side-view/description/
class Solution:
  def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    ans = []

    def dfs(root: Optional[TreeNode], depth: int) -> None:
      if not root:
        return

      if depth == len(ans):
        ans.append(root.val)
      dfs(root.right, depth + 1)
      dfs(root.left, depth + 1)

    dfs(root, 0)
    return ans


# Link: https://leetcode.com/problems/binary-tree-right-side-view/description/
class Solution:
  def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
      return []

    ans = []
    q = collections.deque([root])

    while q:
      size = len(q)
      for i in range(size):
        root = q.popleft()
        if i == size - 1:
          ans.append(root.val)
        if root.left:
          q.append(root.left)
        if root.right:
          q.append(root.right)

    return ans


# Link: https://leetcode.com/problems/next-greater-element-iii/description/
class Solution:
  def nextGreaterElement(self, n: int) -> int:
    def nextPermutation(s: List[chr]) -> str:
      i = len(s) - 2
      while i >= 0:
        if s[i] < s[i + 1]:
          break
        i -= 1

      if i >= 0:
        for j in range(len(s) - 1, i, -1):
          if s[j] > s[i]:
            break
        s[i], s[j] = s[j], s[i]

      reverse(s, i + 1, len(s) - 1)
      return ''.join(s)

    def reverse(s: List[chr], l: int, r: int):
      while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1

    s = nextPermutation(list(str(n)))
    ans = int(s)
    return -1 if ans > 2**31 - 1 or ans <= n else ans


# Link: https://leetcode.com/problems/minimum-sum-of-mountain-triplets-ii/description/
class Solution:
  # Same as 2908. Minimum Sum of Mountain Triplets I
  def minimumSum(self, nums: List[int]) -> int:
    ans = math.inf
    minPrefix = list(itertools.accumulate(nums, min))
    minSuffix = list(itertools.accumulate(reversed(nums), min))[::-1]

    for i, num in enumerate(nums):
      if num > minPrefix[i] and num > minSuffix[i]:
        ans = min(ans, num + minPrefix[i] + minSuffix[i])

    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/sum-of-distances/description/
class Solution:
  def distance(self, nums: List[int]) -> List[int]:
    ans = [0] * len(nums)
    numToIndices = collections.defaultdict(list)

    for i, num in enumerate(nums):
      numToIndices[num].append(i)

    for indices in numToIndices.values():
      n = len(indices)
      if n == 1:
        continue
      sumSoFar = sum(indices)
      prevIndex = 0
      for i in range(n):
        sumSoFar += (i - 1) * (indices[i] - prevIndex)
        sumSoFar -= (n - 1 - i) * (indices[i] - prevIndex)
        ans[indices[i]] = sumSoFar
        prevIndex = indices[i]

    return ans


# Link: https://leetcode.com/problems/minimum-average-difference/description/
class Solution:
  def minimumAverageDifference(self, nums: List[int]) -> int:
    n = len(nums)
    ans = 0
    minDiff = inf
    prefix = 0
    suffix = sum(nums)

    for i, num in enumerate(nums):
      prefix += num
      suffix -= num
      prefixAvg = prefix // (i + 1)
      suffixAvg = 0 if i == n - 1 else suffix // (n - 1 - i)
      diff = abs(prefixAvg - suffixAvg)
      if diff < minDiff:
        ans = i
        minDiff = diff

    return ans


# Link: https://leetcode.com/problems/sorting-three-groups/description/
class Solution:
  def minimumOperations(self, nums: List[int]) -> int:
    # dp[i] := the longest non-decreasing subsequence so far with numbers in [1..i]
    dp = [0] * 4

    for num in nums:
      dp[num] += 1  # Append num to the sequence so far.
      dp[2] = max(dp[2], dp[1])
      dp[3] = max(dp[3], dp[2])

    return len(nums) - dp[3]


# Link: https://leetcode.com/problems/boundary-of-binary-tree/description/
class Solution:
  def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
      return []

    ans = [root.val]

    def dfs(root: Optional[TreeNode], lb: bool, rb: bool):
      """
      1. root.left is left boundary if root is left boundary.
         root.right if left boundary if root.left is None.
      2. Same applys for right boundary.
      3. If root is left boundary, add it before 2 children - preorder.
         If root is right boundary, add it after 2 children - postorder.
      4. A leaf that is neighter left/right boundary belongs to the bottom.
      """
      if not root:
        return
      if lb:
        ans.append(root.val)
      if not lb and not rb and not root.left and not root.right:
        ans.append(root.val)

      dfs(root.left, lb, rb and not root.right)
      dfs(root.right, lb and not root.left, rb)
      if rb:
        ans.append(root.val)

    dfs(root.left, True, False)
    dfs(root.right, False, True)
    return ans


# Link: https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/description/
class Solution:
  def isValidSerialization(self, preorder: str) -> bool:
    degree = 1  # out-degree (children) - in-degree (parent)

    for node in preorder.split(','):
      degree -= 1
      if degree < 0:
        return False
      if node != '#':
        degree += 2

    return degree == 0


# Link: https://leetcode.com/problems/maximum-number-of-people-that-can-be-caught-in-tag/description/
class Solution:
  def catchMaximumAmountofPeople(self, team: List[int], dist: int) -> int:
    ans = 0
    i = 0  # 0s index
    j = 0  # 1s index

    while i < len(team) and j < len(team):
      if i + dist < j or team[i] != 0:
        # Find the next 0 that can be caught by 1.
        i += 1
      elif j + dist < i or team[j] != 1:
        # Find the next 1 that can catch 0.
        j += 1
      else:
        # team[j] catches team[i], so move both.
        ans += 1
        i += 1
        j += 1

    return ans


# Link: https://leetcode.com/problems/maximum-distance-in-arrays/description/
class Solution:
  def maxDistance(self, arrays: List[List[int]]) -> int:
    min1, index_min1 = min((A[0], i) for i, A in enumerate(arrays))
    max1, index_max1 = max((A[-1], i) for i, A in enumerate(arrays))
    if index_min1 != index_max1:
      return max1 - min1

    min2, index_min2 = min((A[0], i)
                           for i, A in enumerate(arrays) if i != index_min1)
    max2, index_min2 = max((A[-1], i)
                           for i, A in enumerate(arrays) if i != index_max1)
    return max(max1 - min2, max2 - min1)


# Link: https://leetcode.com/problems/maximum-distance-in-arrays/description/
class Solution:
  def maxDistance(self, arrays: List[List[int]]) -> int:
    ans = 0
    mini = 10000
    maxi = -10000

    for A in arrays:
      ans = max(ans, A[-1] - mini, maxi - A[0])
      mini = min(mini, A[0])
      maxi = max(maxi, A[-1])

    return ans


# Link: https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/description/
class Solution:
  def minimumRounds(self, tasks: List[int]) -> int:
    freqs = collections.Counter(tasks).values()
    return -1 if 1 in freqs else sum((f + 2) // 3 for f in freqs)


# Link: https://leetcode.com/problems/search-a-2d-matrix-ii/description/
class Solution:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    r = 0
    c = len(matrix[0]) - 1

    while r < len(matrix) and c >= 0:
      if matrix[r][c] == target:
        return True
      if target < matrix[r][c]:
        c -= 1
      else:
        r += 1

    return False


# Link: https://leetcode.com/problems/count-square-submatrices-with-all-ones/description/
class Solution:
  def countSquares(self, matrix: List[List[int]]) -> int:
    for i in range(len(matrix)):
      for j in range(len(matrix[0])):
        if matrix[i][j] == 1 and i > 0 and j > 0:
          matrix[i][j] += min(matrix[i - 1][j - 1],
                              matrix[i - 1][j], matrix[i][j - 1])
    return sum(sum(row) for row in matrix)


# Link: https://leetcode.com/problems/build-an-array-with-stack-operations/description/
class Solution:
  def buildArray(self, target: List[int], n: int) -> List[str]:
    ans = []
    i = 0  # Target pointer
    num = 1  # Curr num

    while i < len(target):
      t = target[i]
      if t == num:
        ans.append("Push")
        i += 1
      else:
        ans.append("Push")
        ans.append("Pop")
      num += 1

    return ans


# Link: https://leetcode.com/problems/path-sum-iii/description/
class Solution:
  def pathSum(self, root: TreeNode, summ: int) -> int:
    if not root:
      return 0

    def dfs(root: TreeNode, summ: int) -> int:
      if not root:
        return 0
      return (summ == root.val) + \
          dfs(root.left, summ - root.val) + \
          dfs(root.right, summ - root.val)

    return dfs(root, summ) + \
        self.pathSum(root.left, summ) + \
        self.pathSum(root.right, summ)


# Link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/
class Solution:
  def connect(self, root: 'Node') -> 'Node':
    node = root  # the node that is above the current needling

    while node and node.left:
      dummy = Node(0)  # a dummy node before needling
      # Needle the children of the node.
      needle = dummy
      while node:
        needle.next = node.left
        needle = needle.next
        needle.next = node.right
        needle = needle.next
        node = node.next
      node = dummy.next  # Move the node to the next level.

    return root


# Link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/
class Solution:
  def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
      return None

    def connectTwoNodes(p, q) -> None:
      if not p:
        return
      p.next = q
      connectTwoNodes(p.left, p.right)
      connectTwoNodes(q.left, q.right)
      connectTwoNodes(p.right, q.left)

    connectTwoNodes(root.left, root.right)
    return root


# Link: https://leetcode.com/problems/next-closest-time/description/
class Solution:
  def nextClosestTime(self, time: str) -> str:
    ans = list(time)
    digits = sorted(ans)

    def nextClosest(digit: str, limit: str) -> chr:
      next = bisect_right(digits, digit)
      return digits[0] if next == 4 or digits[next] > limit else digits[next]

    ans[4] = nextClosest(ans[4], '9')
    if time[4] < ans[4]:
      return ''.join(ans)

    ans[3] = nextClosest(ans[3], '5')
    if time[3] < ans[3]:
      return ''.join(ans)

    ans[1] = nextClosest(ans[1], '3' if ans[0] == '2' else '9')
    if time[1] < ans[1]:
      return ''.join(ans)

    ans[0] = nextClosest(ans[0], '2')
    return ''.join(ans)


# Link: https://leetcode.com/problems/longest-arithmetic-subsequence/description/
class Solution:
  def longestArithSeqLength(self, nums: List[int]) -> int:
    n = len(nums)
    ans = 0
    # dp[i][k] := the length of the longest arithmetic subsequence of nums[0..i]
    # with k = diff + 500
    dp = [[0] * 1001 for _ in range(n)]

    for i in range(n):
      for j in range(i):
        k = nums[i] - nums[j] + 500
        dp[i][k] = max(2, dp[j][k] + 1)
        ans = max(ans, dp[i][k])

    return ans


# Link: https://leetcode.com/problems/insufficient-nodes-in-root-to-leaf-paths/description/
class Solution:
  def sufficientSubset(self, root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:
    if not root:
      return None
    if not root.left and not root.right:
      return None if root.val < limit else root
    root.left = self.sufficientSubset(root.left, limit - root.val)
    root.right = self.sufficientSubset(root.right, limit - root.val)
    return None if not root.left and not root.right else root


# Link: https://leetcode.com/problems/disconnect-path-in-a-binary-matrix-by-at-most-one-flip/description/
class Solution:
  def isPossibleToCutPath(self, grid: List[List[int]]) -> bool:
    # Returns True is there's a path from (0, 0) to (m - 1, n - 1).
    # Also marks the visited path as 0 except (m - 1, n - 1).
    def hasPath(i: int, j: int) -> bool:
      if i == len(grid) or j == len(grid[0]):
        return False
      if i == len(grid) - 1 and j == len(grid[0]) - 1:
        return True
      if grid[i][j] == 0:
        return False

      grid[i][j] = 0
      # Go down first. Since we use OR logic, we'll only mark one path.
      return hasPath(i + 1, j) or hasPath(i, j + 1)

    if not hasPath(0, 0):
      return True
    # Reassign (0, 0) as 1.
    grid[0][0] = 1
    return not hasPath(0, 0)


# Link: https://leetcode.com/problems/beautiful-towers-ii/description/
class Solution:
  # Same as 2865. Beautiful Towers I
  def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
    n = len(maxHeights)
    maxSum = [0] * n  # maxSum[i] := the maximum sum with peak i

    def process(stack: List[int], i: int, summ: int) -> int:
      while len(stack) > 1 and maxHeights[stack[-1]] > maxHeights[i]:
        j = stack.pop()
        # The last abs(j - stack[-1]) heights are maxHeights[j].
        summ -= abs(j - stack[-1]) * maxHeights[j]
      # Put abs(i - stack[-1]) `maxHeight` in heights.
      summ += abs(i - stack[-1]) * maxHeights[i]
      stack.append(i)
      return summ

    stack = [-1]
    summ = 0
    for i in range(len(maxHeights)):
      summ = process(stack, i, summ)
      maxSum[i] = summ

    stack = [n]
    summ = 0
    for i in range(n - 1, -1, -1):
      summ = process(stack, i, summ)
      maxSum[i] += summ - maxHeights[i]

    return max(maxSum)


# Link: https://leetcode.com/problems/swap-adjacent-in-lr-string/description/
class Solution:
  def canTransform(self, start: str, end: str) -> bool:
    if start.replace('X', '') != end.replace('X', ''):
      return False

    i = 0  # start's index
    j = 0  # end's index

    while i < len(start) and j < len(end):
      while i < len(start) and start[i] == 'X':
        i += 1
      while j < len(end) and end[j] == 'X':
        j += 1
      if i == len(start) and j == len(end):
        return True
      if i == len(start) or j == len(end):
        return False
      # L can only move to left.
      if start[i] == 'L' and i < j:
        return False
      # R can only move to right.
      if start[i] == 'R' and i > j:
        return False
      i += 1
      j += 1

    return True


# Link: https://leetcode.com/problems/binary-tree-vertical-order-traversal/description/
class Solution:
  def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
      return []

    range_ = [0] * 2

    def getRange(root: Optional[TreeNode], x: int) -> None:
      if not root:
        return

      range_[0] = min(range_[0], x)
      range_[1] = max(range_[1], x)

      getRange(root.left, x - 1)
      getRange(root.right, x + 1)

    getRange(root, 0)  # Get the leftmost and the rightmost x index.

    ans = [[] for _ in range(range_[1] - range_[0] + 1)]
    q = collections.deque([(root, -range_[0])])  # (TreeNode, x)

    while q:
      node, x = q.popleft()
      ans[x].append(node.val)
      if node.left:
        q.append((node.left, x - 1))
      if node.right:
        q.append((node.right, x + 1))

    return ans


# Link: https://leetcode.com/problems/valid-triangle-number/description/
class Solution:
  def triangleNumber(self, nums: List[int]) -> int:
    ans = 0

    nums.sort()

    for k in range(len(nums) - 1, 1, -1):
      i = 0
      j = k - 1
      while i < j:
        if nums[i] + nums[j] > nums[k]:
          ans += j - i
          j -= 1
        else:
          i += 1

    return ans


# Link: https://leetcode.com/problems/3sum-smaller/description/
class Solution:
  def threeSumSmaller(self, nums: List[int], target: int) -> int:
    if len(nums) < 3:
      return 0

    ans = 0

    nums.sort()

    for i in range(len(nums) - 2):
      l = i + 1
      r = len(nums) - 1
      while l < r:
        if nums[i] + nums[l] + nums[r] < target:
          # (nums[i], nums[l], nums[r])
          # (nums[i], nums[l], nums[r - 1])
          # ...,
          # (nums[i], nums[l], nums[l + 1])
          ans += r - l
          l += 1
        else:
          r -= 1

    return ans


# Link: https://leetcode.com/problems/find-k-closest-elements/description/
class Solution:
  def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    l = 0
    r = len(arr) - k

    while l < r:
      m = (l + r) // 2
      if x - arr[m] <= arr[m + k] - x:
        r = m
      else:
        l = m + 1

    return arr[l:l + k]


# Link: https://leetcode.com/problems/all-paths-from-source-to-target/description/
class Solution:
  def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    ans = []

    def dfs(u: int, path: List[int]) -> None:
      if u == len(graph) - 1:
        ans.append(path)
        return

      for v in graph[u]:
        dfs(v, path + [v])

    dfs(0, [0])
    return ans


# Link: https://leetcode.com/problems/nested-list-weight-sum-ii/description/
class Solution:
  def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
    ans = 0
    prevSum = 0
    q = collections.deque(nestedList)

    while q:
      for _ in range(len(q)):
        ni = q.popleft()
        if ni.isInteger():
          prevSum += ni.getInteger()
        else:
          for nextNi in ni.getList():
            q.append(nextNi)
      ans += prevSum

    return ans


# Link: https://leetcode.com/problems/stone-game-ix/description/
class Solution:
  def stoneGameIX(self, stones: List[int]) -> bool:
    count = collections.Counter(stone % 3 for stone in stones)
    if count[0] % 2 == 0:
      return min(count[1], count[2]) > 0
    return abs(count[1] - count[2]) > 2


# Link: https://leetcode.com/problems/valid-sudoku/description/
class Solution:
  def isValidSudoku(self, board: List[List[str]]) -> bool:
    seen = set()

    for i in range(9):
      for j in range(9):
        c = board[i][j]
        if c == '.':
          continue
        if c + '@row ' + str(i) in seen or \
           c + '@col ' + str(j) in seen or \
           c + '@box ' + str(i // 3) + str(j // 3) in seen:
          return False
        seen.add(c + '@row ' + str(i))
        seen.add(c + '@col ' + str(j))
        seen.add(c + '@box ' + str(i // 3) + str(j // 3))

    return True


# Link: https://leetcode.com/problems/maximum-of-minimum-values-in-all-subarrays/description/
class Solution:
  # Similar to 1950. Maximum of Minimum Values in All Subarrays
  def findMaximums(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = [0] * n
    # prevMin[i] := the index k s.t.
    # nums[k] is the previous minimum in nums[0..n)
    prevMin = [-1] * n
    # nextMin[i] := the index k s.t.
    # nums[k] is the next minimum innums[i + 1..n)
    nextMin = [n] * n
    stack = []

    for i, num in enumerate(nums):
      while stack and nums[stack[-1]] > nums[i]:
        index = stack.pop()
        nextMin[index] = i
      if stack:
        prevMin[i] = stack[-1]
      stack.append(i)

    # For each nums[i], let l = nextMin[i] + 1 and r = nextMin[i] - 1.
    # nums[i] is the minimun in nums[l..r].
    # So, the ans[r - l + 1] will be at least nums[i].
    for num, l, r in zip(nums, prevMin, nextMin):
      sz = r - l - 1
      ans[sz - 1] = max(ans[sz - 1], num)

    # ans[i] should always >= ans[i + 1..n).
    for i in range(n - 2, -1, -1):
      ans[i] = max(ans[i], ans[i + 1])

    return ans


# Link: https://leetcode.com/problems/maximum-trailing-zeros-in-a-cornered-path/description/
class Solution:
  def maxTrailingZeros(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    # leftPrefix2[i][j] := the number of 2 in grid[i][0..j]
    # leftPrefix5[i][j] := the number of 5 in grid[i][0..j]
    # topPrefix2[i][j] := the number of 2 in grid[0..i][j]
    # topPrefix5[i][j] := the number of 5 in grid[0..i][j]
    leftPrefix2 = [[0] * n for _ in range(m)]
    leftPrefix5 = [[0] * n for _ in range(m)]
    topPrefix2 = [[0] * n for _ in range(m)]
    topPrefix5 = [[0] * n for _ in range(m)]

    def getCount(num: int, factor: int) -> int:
      count = 0
      while num % factor == 0:
        num //= factor
        count += 1
      return count

    for i in range(m):
      for j in range(n):
        leftPrefix2[i][j] = getCount(grid[i][j], 2)
        leftPrefix5[i][j] = getCount(grid[i][j], 5)
        if j:
          leftPrefix2[i][j] += leftPrefix2[i][j - 1]
          leftPrefix5[i][j] += leftPrefix5[i][j - 1]

    for j in range(n):
      for i in range(m):
        topPrefix2[i][j] = getCount(grid[i][j], 2)
        topPrefix5[i][j] = getCount(grid[i][j], 5)
        if i:
          topPrefix2[i][j] += topPrefix2[i - 1][j]
          topPrefix5[i][j] += topPrefix5[i - 1][j]

    ans = 0
    for i in range(m):
      for j in range(n):
        curr2 = getCount(grid[i][j], 2)
        curr5 = getCount(grid[i][j], 5)
        l2 = leftPrefix2[i][j]
        l5 = leftPrefix5[i][j]
        r2 = leftPrefix2[i][n - 1] - (0 if j == 0 else leftPrefix2[i][j - 1])
        r5 = leftPrefix5[i][n - 1] - (0 if j == 0 else leftPrefix5[i][j - 1])
        t2 = topPrefix2[i][j]
        t5 = topPrefix5[i][j]
        d2 = topPrefix2[m - 1][j] - (0 if i == 0 else topPrefix2[i - 1][j])
        d5 = topPrefix5[m - 1][j] - (0 if i == 0 else topPrefix5[i - 1][j])
        ans = max(ans,
                  min(l2 + t2 - curr2, l5 + t5 - curr5),
                  min(r2 + t2 - curr2, r5 + t5 - curr5),
                  min(l2 + d2 - curr2, l5 + d5 - curr5),
                  min(r2 + d2 - curr2, r5 + d5 - curr5))

    return ans


# Link: https://leetcode.com/problems/shifting-letters/description/
class Solution:
  def shiftingLetters(self, s: str, shifts: List[int]) -> str:
    ans = []

    for i in reversed(range(len(shifts) - 1)):
      shifts[i] += shifts[i + 1]

    for c, shift in zip(s, shifts):
      ans.append(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))

    return ''.join(ans)


# Link: https://leetcode.com/problems/separate-black-and-white-balls/description/
class Solution:
  def minimumSteps(self, s: str) -> int:
    ans = 0
    ones = 0

    for c in s:
      if c == '1':
        ones += 1
      else:  # Move 1s to the front of the current '0'.
        ans += ones

    return ans


# Link: https://leetcode.com/problems/movement-of-robots/description/
class Solution:
  def sumDistance(self, nums: List[int], s: str, d: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    prefix = 0
    pos = sorted([num - d if c == 'L' else num + d
                  for num, c in zip(nums, s)])

    for i, p in enumerate(pos):
      ans = ((ans + i * p - prefix) % kMod + kMod) % kMod
      prefix = ((prefix + p) % kMod + kMod) % kMod

    return ans


# Link: https://leetcode.com/problems/minimize-xor/description/
class Solution:
  def minimizeXor(self, num1: int, num2: int) -> int:
    kMaxBit = 30
    bits = num2.bit_count()
    # Can turn off all the bits in `num1`.
    if num1.bit_count() == bits:
      return num1

    ans = 0

    # Turn off the MSB if we have `bits` quota.
    for i in reversed(range(kMaxBit)):
      if num1 >> i & 1:
        ans |= 1 << i
        bits -= 1
        if bits == 0:
          return ans

    # Turn on the LSB if we still have `bits`.
    for i in range(kMaxBit):
      if (num1 >> i & 1) == 0:
        ans |= 1 << i
        bits -= 1
        if bits == 0:
          return ans

    return ans


# Link: https://leetcode.com/problems/ones-and-zeroes/description/
class Solution:
  def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
    # dp[i][j] := the maximum size of the subset given i 0s and j 1s are
    # available
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for s in strs:
      count0 = s.count('0')
      count1 = len(s) - count0
      for i in range(m, count0 - 1, -1):
        for j in range(n, count1 - 1, -1):
          dp[i][j] = max(dp[i][j], dp[i - count0][j - count1] + 1)

    return dp[m][n]


# Link: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
class Solution:
  def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    inToIndex = {num: i for i, num in enumerate(inorder)}

    def build(preStart: int, preEnd: int, inStart: int, inEnd: int) -> Optional[TreeNode]:
      if preStart > preEnd:
        return None

      rootVal = preorder[preStart]
      rootInIndex = inToIndex[rootVal]
      leftSize = rootInIndex - inStart

      root = TreeNode(rootVal)
      root.left = build(preStart + 1, preStart + leftSize,
                        inStart, rootInIndex - 1)
      root.right = build(preStart + leftSize + 1,
                         preEnd, rootInIndex + 1, inEnd)
      return root

    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


# Link: https://leetcode.com/problems/insert-delete-getrandom-o1/description/
class RandomizedSet:
  def __init__(self):
    self.vals = []
    self.valToIndex = collections.defaultdict(int)  # {val: index in vals}

  def insert(self, val: int) -> bool:
    if val in self.valToIndex:
      return False
    self.valToIndex[val] = len(self.vals)
    self.vals.append(val)
    return True

  def remove(self, val: int) -> bool:
    if val not in self.valToIndex:
      return False
    index = self.valToIndex[val]
    # The order of the following two lines is important when vals.size() == 1.
    self.valToIndex[self.vals[-1]] = index
    del self.valToIndex[val]
    self.vals[index] = self.vals[-1]
    self.vals.pop()
    return True

  def getRandom(self) -> int:
    index = random.randint(0, len(self.vals) - 1)
    return self.vals[index]


# Link: https://leetcode.com/problems/sum-of-absolute-differences-in-a-sorted-array/description/
class Solution:
  def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
    prefix = list(itertools.accumulate(nums))
    suffix = list(itertools.accumulate(nums[::-1]))[::-1]
    return [num * (i + 1) - prefix[i] + suffix[i] - num * (len(nums) - i)
            for i, num in enumerate(nums)]


# Link: https://leetcode.com/problems/generalized-abbreviation/description/
class Solution:
  def generateAbbreviations(self, word: str) -> List[str]:
    ans = []

    def getCountString(count: int) -> str:
      return str(count) if count > 0 else ''

    def dfs(i: int, count: int, path: List[str]) -> None:
      if i == len(word):
        ans.append(''.join(path) + getCountString(count))
        return

      # Abbreviate the word[i].
      dfs(i + 1, count + 1, path)
      # Keep the word[i], so consume the count as a string.
      path.append(getCountString(count) + word[i])
      # Reset the count to 0.
      dfs(i + 1, 0, path)
      path.pop()

    dfs(0, 0, [])
    return ans


# Link: https://leetcode.com/problems/count-the-number-of-houses-at-a-certain-distance-i/description/
class Solution:
  def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
    if x > y:
      x, y = y, x

    def bothInRing(ringLen: int) -> List[int]:
      """
      Returns the contribution from the scenario where two houses are located
      in the ring.
      """
      res = [0] * n
      for k in range(1, (ringLen - 1) // 2 + 1):
        res[k - 1] += ringLen
      if ringLen % 2 == 0:
        res[ringLen // 2 - 1] += ringLen // 2
      return res

    def bothInTheSameLine(lineLen: int) -> List[int]:
      """
      Returns the contribution from the scenario where two houses are either
      located in the left line [1, x) or the right line (y, n].
      """
      res = [0] * n
      for k in range(1, lineLen + 1):
        res[k - 1] += lineLen - k
      return res

    def lineToRing(lineLen: int, ringLen: int) -> List[int]:
      """
      Returns the contribution from the scenario where one house is either
      located in the left line [1, x) or the right line (y, n] and the
      other house is located in the cycle.
      """
      res = [0] * n
      for k in range(1, lineLen + ringLen):
        # min(
        #   at most k - 1 since we need to give 1 to the line,
        #   at most ringLen / 2 since for length > ringLen / 2, it can always be
        #     calculated as ringLen - ringLen / 2
        # )
        maxInRingLen = min(k - 1, ringLen // 2)
        # max(at least 0, at lest k - lineLen)
        minInRingLen = max(0, k - lineLen)
        if minInRingLen <= maxInRingLen:
          # Each ring length contributes 2 to the count due to the split of
          # paths when entering the ring: One path traverses the upper half of
          # the ring, and the other traverses the lower half.
          # This is illustrated as follows:
          #   Path 1: ... -- x -- (upper half of the ring)
          #   Path 2: ... -- x -- (lower half of the ring)
          res[k - 1] += (maxInRingLen - minInRingLen + 1) * 2
          if minInRingLen == 0:
            # Subtract 1 since there's no split.
            res[k - 1] -= 1
          if maxInRingLen * 2 == ringLen:
            # Subtract 1 since the following case only contribute one:
            #   ... -- x -- (upper half of the ring) -- middle point
            #   ... -- x -- (upper half of the ring) -- middle point
            res[k - 1] -= 1
      return res

    def lineToLine(leftLineLen: int, rightLineLen: int) -> List[int]:
      """
      Returns the contribution from the scenario where one house is in the left
      line [1, x) and the other house is in the right line (y, n].
      """
      res = [0] * n
      for k in range(leftLineLen + rightLineLen + 2):
        # min(
        #   at most leftLineLen,
        #   at most k - 1 - (x < y) since we need to give 1 to the right line
        #     and if x < y we need to give another 1 to "x - y".
        # )
        maxInLeft = min(leftLineLen, k - 1 - (x < y))
        # max(at least 1, at least k - rightLineLen - (x < y))
        minInLeft = max(1, k - rightLineLen - (x < y))
        if minInLeft <= maxInLeft:
          res[k - 1] += maxInLeft - minInLeft + 1
      return res

    ringLen = y - x + 1
    leftLineLen = x - 1
    rightLineLen = (n - y)

    ans = [0] * n
    ans = list(map(operator.add, ans, bothInRing(ringLen)))
    ans = list(map(operator.add, ans, bothInTheSameLine(leftLineLen)))
    ans = list(map(operator.add, ans, bothInTheSameLine(rightLineLen)))
    ans = list(map(operator.add, ans, lineToRing(leftLineLen, ringLen)))
    ans = list(map(operator.add, ans, lineToRing(rightLineLen, ringLen)))
    ans = list(map(operator.add, ans, lineToLine(leftLineLen, rightLineLen)))
    return [freq * 2 for freq in ans]


# Link: https://leetcode.com/problems/shortest-path-to-get-food/description/
class Solution:
  def getFood(self, grid: List[List[str]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    ans = 0
    q = collections.deque([self._getStartLocation(grid)])

    while q:
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if grid[x][y] == 'X':
            continue
          if grid[x][y] == '#':
            return ans + 1
          q.append((x, y))
          grid[x][y] = 'X'  # Mark as visited.
      ans += 1

    return -1

  def _getStartLocation(self, grid: List[List[str]]) -> Tuple[int, int]:
    for i, row in enumerate(grid):
      for j, cell in enumerate(row):
        if cell == '*':
          return (i, j)


# Link: https://leetcode.com/problems/maximum-number-of-removable-characters/description/
class Solution:
  def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
    l = 0
    r = len(removable) + 1

    def remove(k: int) -> str:
      removed = [c for c in s]
      for i in range(k):
        removed[removable[i]] = '*'
      return ''.join(removed)

    def isSubsequence(p: str, s: str) -> bool:
      i = 0
      for j, c in enumerate(s):
        if p[i] == s[j]:
          i += 1
          if i == len(p):
            return True
      return False

    while l < r:
      m = (l + r) // 2
      removed = remove(m)
      if isSubsequence(p, removed):
        l = m + 1
      else:
        r = m

    return l - 1


# Link: https://leetcode.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/description/
class Solution:
  def minPartitions(self, n: str) -> int:
    return int(max(n))


# Link: https://leetcode.com/problems/design-authentication-manager/description/
class AuthenticationManager:
  def __init__(self, timeToLive: int):
    self.timeToLive = timeToLive
    self.tokenIdToExpiryTime = {}
    self.times = collections.defaultdict(int)

  def generate(self, tokenId: str, currentTime: int) -> None:
    self.tokenIdToExpiryTime[tokenId] = currentTime
    self.times[currentTime] += 1

  def renew(self, tokenId: str, currentTime: int) -> None:
    expiryTime = self.tokenIdToExpiryTime.get(tokenId)
    if expiryTime is None or currentTime >= expiryTime + self.timeToLive:
      return
    del self.times[expiryTime]
    self.tokenIdToExpiryTime[tokenId] = currentTime
    self.times[currentTime] += 1

  def countUnexpiredTokens(self, currentTime: int) -> int:
    expiryTimeThreshold = currentTime - self.timeToLive + 1
    for expiryTime in list(self.times.keys()):
      if expiryTime < expiryTimeThreshold:
        del self.times[expiryTime]
    return sum(self.times.values())


# Link: https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/description/
class Solution:
  def findMaxFish(self, grid: List[List[int]]) -> int:
    def dfs(i: int, j: int) -> int:
      if i < 0 or i == len(grid) or j < 0 or j == len(grid[0]):
        return 0
      if grid[i][j] == 0:
        return 0
      caughtFish = grid[i][j]
      grid[i][j] = 0  # Mark 0 as visited
      return caughtFish + \
          dfs(i + 1, j) + dfs(i - 1, j) + \
          dfs(i, j + 1) + dfs(i, j - 1)

    return max(dfs(i, j)
               for i in range(len(grid))
               for j in range(len(grid[0])))


# Link: https://leetcode.com/problems/closest-prime-numbers-in-range/description/
class Solution:
  def closestPrimes(self, left: int, right: int) -> List[int]:
    isPrime = self._sieveEratosthenes(right + 1)
    primes = [i for i in range(left, right + 1) if isPrime[i]]

    if len(primes) < 2:
      return [-1, -1]

    minDiff = math.inf
    num1 = -1
    num2 = -1

    for a, b in zip(primes, primes[1:]):
      diff = b - a
      if diff < minDiff:
        minDiff = diff
        num1 = a
        num2 = b

    return [num1, num2]

  def _sieveEratosthenes(self, n: int) -> List[bool]:
    isPrime = [True] * n
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n**0.5) + 1):
      if isPrime[i]:
        for j in range(i * i, n, i):
          isPrime[j] = False
    return isPrime


# Link: https://leetcode.com/problems/split-two-strings-to-make-palindrome/description/
class Solution:
  def checkPalindromeFormation(self, a: str, b: str) -> bool:
    return self._check(a, b) or self._check(b, a)

  def _check(self, a: str, b: str) -> bool:
    i, j = 0, len(a) - 1
    while i < j:
      if a[i] != b[j]:
        # a[0:i] + a[i..j] + b[j + 1:] or
        # a[0:i] + b[i..j] + b[j + 1:]
        return self._isPalindrome(a, i, j) or self._isPalindrome(b, i, j)
      i += 1
      j -= 1
    return True

  def _isPalindrome(self, s: str, i: int, j: int) -> bool:
    while i < j:
      if s[i] != s[j]:
        return False
      i += 1
      j -= 1
    return True


# Link: https://leetcode.com/problems/split-a-circular-linked-list/description/
class Solution:
  def splitCircularLinkedList(self, list: Optional[ListNode]) -> List[Optional[ListNode]]:
    slow = list
    fast = list

    # Point `slow` to the last node in the first half.
    while fast.next != list and fast.next.next != list:
      slow = slow.next
      fast = fast.next.next

    # Circle back the second half.
    secondHead = slow.next
    if fast.next == list:
      fast.next = secondHead
    else:
      fast.next.next = secondHead

    # Circle back the first half.
    slow.next = list

    return [list, secondHead]


# Link: https://leetcode.com/problems/valid-square/description/
class Solution:
  def validSquare(self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
    def dist(p1: List[int], p2: List[int]) -> int:
      return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    distSet = set([dist(*pair)
                   for pair in list(itertools.combinations([p1, p2, p3, p4], 2))])

    return 0 not in distSet and len(distSet) == 2


# Link: https://leetcode.com/problems/jump-game-vii/description/
class Solution:
  def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
    count = 0
    dp = [True] + [False] * (len(s) - 1)

    for i in range(minJump, len(s)):
      count += dp[i - minJump]
      if i - maxJump > 0:
        count -= dp[i - maxJump - 1]
      dp[i] = count and s[i] == '0'

    return dp[-1]


# Link: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
class Solution:
  def findMin(self, nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1

    while l < r:
      m = (l + r) // 2
      if nums[m] < nums[r]:
        r = m
      else:
        l = m + 1

    return nums[l]


# Link: https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/description/
class Solution:
  def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(maze)
    n = len(maze[0])
    ans = 0
    q = collections.deque([(entrance[0], entrance[1])])
    seen = {(entrance[0], entrance[1])}

    while q:
      ans += 1
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if (x, y) in seen or maze[x][y] == '+':
            continue
          if x == 0 or x == m - 1 or y == 0 or y == n - 1:
            return ans
          q.append((x, y))
          seen.add((x, y))

    return -1


# Link: https://leetcode.com/problems/subsets-ii/description/
class Solution:
  def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    ans = []

    def dfs(s: int, path: List[int]) -> None:
      ans.append(path)
      if s == len(nums):
        return

      for i in range(s, len(nums)):
        if i > s and nums[i] == nums[i - 1]:
          continue
        dfs(i + 1, path + [nums[i]])

    nums.sort()
    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts/description/
class Solution:
  def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
    kMod = 1_000_000_007
    # the maximum gap of each direction
    maxGapX = max(b - a
                  for a, b in itertools.pairwise([0] + sorted(horizontalCuts) + [h]))
    maxGapY = max(b - a
                  for a, b in itertools.pairwise([0] + sorted(verticalCuts) + [w]))
    return maxGapX * maxGapY % kMod


# Link: https://leetcode.com/problems/maximum-average-subtree/description/
class T:
  def __init__(self, summ: int, count: int, maxAverage: float):
    self.summ = summ
    self.count = count
    self.maxAverage = maxAverage


class Solution:
  def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
    def maximumAverage(root: Optional[TreeNode]) -> T:
      if not root:
        return T(0, 0, 0)

      left = maximumAverage(root.left)
      right = maximumAverage(root.right)

      summ = root.val + left.summ + right.summ
      count = 1 + left.count + right.count
      maxAverage = max(summ / count, left.maxAverage, right.maxAverage)
      return T(summ, count, maxAverage)

    return maximumAverage(root).maxAverage


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-array-empty/description/
class Solution:
  def minOperations(self, nums: List[int]) -> int:
    count = collections.Counter(nums)
    if 1 in count.values():
      return -1
    return sum((freq + 2) // 3 for freq in count.values())


# Link: https://leetcode.com/problems/jump-game/description/
class Solution:
  def canJump(self, nums: List[int]) -> bool:
    i = 0
    reach = 0

    while i < len(nums) and i <= reach:
      reach = max(reach, i + nums[i])
      i += 1

    return i == len(nums)


# Link: https://leetcode.com/problems/put-boxes-into-the-warehouse-i/description/
class Solution:
  def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
    realWarehouse = [warehouse[0]]

    for i in range(1, len(warehouse)):
      realWarehouse.append(min(realWarehouse[-1], warehouse[i]))

    boxes.sort()
    i = 0  # boxes' index
    for height in reversed(realWarehouse):
      if i < len(boxes) and boxes[i] <= height:
        i += 1

    return i


# Link: https://leetcode.com/problems/put-boxes-into-the-warehouse-i/description/
class Solution:
  def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
    i = 0  # warehouse's index

    for box in sorted(boxes, reverse=True):
      if i < len(warehouse) and warehouse[i] >= box:
        i += 1

    return i


# Link: https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/description/
class Solution:
  def getWordsInLongestSubsequence(self, n: int, words: List[str], groups: List[int]) -> List[str]:
    ans = []
    # dp[i] := the length of the longest subsequence ending in `words[i]`
    dp = [1] * n
    # prev[i] := the best index of words[i]
    prev = [-1] * n

    for i in range(1, n):
      for j in range(i):
        if groups[i] == groups[j]:
          continue
        if len(words[i]) != len(words[j]):
          continue
        if sum(a != b for a, b in zip(words[i], words[j])) != 1:
          continue
        if dp[i] < dp[j] + 1:
          dp[i] = dp[j] + 1
          prev[i] = j

    # Find the last index of the subsequence.
    index = dp.index(max(dp))
    while index != -1:
      ans.append(words[index])
      index = prev[index]

    return ans[::-1]


# Link: https://leetcode.com/problems/asteroid-collision/description/
class Solution:
  def asteroidCollision(self, asteroids: List[int]) -> List[int]:
    stack = []

    for a in asteroids:
      if a > 0:
        stack.append(a)
      else:  # a < 0
        # Destroy the previous positive one(s).
        while stack and stack[-1] > 0 and stack[-1] < -a:
          stack.pop()
        if not stack or stack[-1] < 0:
          stack.append(a)
        elif stack[-1] == -a:
          stack.pop()  # Both asteroids explode.
        else:  # stack[-1] > the current asteroid.
          pass  # Destroy the current asteroid, so do nothing.

    return stack


# Link: https://leetcode.com/problems/the-kth-factor-of-n/description/
class Solution:
  def kthFactor(self, n: int, k: int) -> int:
    # If i is a divisor of n, then n // i is also a divisor of n. So, we can
    # find all the divisors of n by processing the numbers <= sqrt(n).
    factor = 1
    i = 0  # the i-th factor

    while factor < int(math.sqrt(n)):
      if n % factor == 0:
        i += 1
        if i == k:
          return factor
      factor += 1

    factor = n // factor
    while factor >= 1:
      if n % factor == 0:
        i += 1
        if i == k:
          return n // factor
      factor -= 1

    return -1


# Link: https://leetcode.com/problems/longest-non-decreasing-subarray-from-two-arrays/description/
class Solution:
  def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
    ans = 1
    dp1 = 1  # the longest subarray that ends in nums1[i] so far
    dp2 = 1  # the longest subarray that ends in nums2[i] so far

    for i in range(1, len(nums1)):
      dp11 = dp1 + 1 if nums1[i - 1] <= nums1[i] else 1
      dp21 = dp2 + 1 if nums2[i - 1] <= nums1[i] else 1
      dp12 = dp1 + 1 if nums1[i - 1] <= nums2[i] else 1
      dp22 = dp2 + 1 if nums2[i - 1] <= nums2[i] else 1
      dp1 = max(dp11, dp21)
      dp2 = max(dp12, dp22)
      ans = max(ans, dp1, dp2)

    return ans


# Link: https://leetcode.com/problems/number-of-adjacent-elements-with-the-same-color/description/
class Solution:
  def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
    ans = []
    arr = [0] * n
    sameColors = 0

    for i, color in queries:
      if i + 1 < n:
        if arr[i + 1] > 0 and arr[i + 1] == arr[i]:
          sameColors -= 1
        if arr[i + 1] == color:
          sameColors += 1
      if i > 0:
        if arr[i - 1] > 0 and arr[i - 1] == arr[i]:
          sameColors -= 1
        if arr[i - 1] == color:
          sameColors += 1
      arr[i] = color
      ans.append(sameColors)

    return ans


# Link: https://leetcode.com/problems/find-xor-beauty-of-array/description/
class Solution:
  def xorBeauty(self, nums: List[int]) -> int:
    return functools.reduce(operator.xor, nums)


# Link: https://leetcode.com/problems/maximum-number-of-coins-you-can-get/description/
class Solution:
  def maxCoins(self, piles: List[int]) -> int:
    return sum(sorted(piles)[len(piles) // 3::2])


# Link: https://leetcode.com/problems/number-of-ways-to-split-array/description/
class Solution:
  def waysToSplitArray(self, nums: List[int]) -> int:
    ans = 0
    prefix = 0
    suffix = sum(nums)

    for i in range(len(nums) - 1):
      prefix += nums[i]
      suffix -= nums[i]
      if prefix >= suffix:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/count-the-number-of-square-free-subsets/description/
class Solution:
  def squareFreeSubsets(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def getMask(num: int) -> int:
      """
      e.g. num = 10 = 2 * 5, so mask = 0b101 . 0b1010 (append a 0)
           num = 15 = 3 * 5, so mask = 0b110 . 0b1100 (append a 0)
           num = 25 = 5 * 5, so mask =  (-1)2 . (1..1)2 (invalid)
      """
      mask = 0
      for i, prime in enumerate(primes):
        rootCount = 0
        while num % prime == 0:
          num //= prime
          rootCount += 1
        if rootCount >= 2:
          return -1
        if rootCount == 1:
          mask |= 1 << i
      return mask << 1

    masks = [getMask(num) for num in nums]

    @functools.lru_cache(None)
    def dp(i: int, used: int) -> int:
      if i == len(masks):
        return 1
      pick = dp(i + 1, used | masks[i]) if (masks[i] & used) == 0 else 0
      skip = dp(i + 1, used)
      return (pick + skip) % kMod

    # -1 means that we take no number.
    # `used` is initialized to 1 so that -1 & 1 = 1 instead of 0.
    return (dp(0, 1) - 1 + kMod) % kMod


# Link: https://leetcode.com/problems/longest-nice-subarray/description/
class Solution:
  def longestNiceSubarray(self, nums: List[int]) -> int:
    ans = 0
    used = 0

    l = 0
    for r, num in enumerate(nums):
      while used & num:
        used ^= nums[l]
        l += 1
      used |= num
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/video-stitching/description/
class Solution:
  def videoStitching(self, clips: List[List[int]], time: int) -> int:
    ans = 0
    end = 0
    farthest = 0

    clips.sort()

    i = 0
    while farthest < time:
      while i < len(clips) and clips[i][0] <= end:
        farthest = max(farthest, clips[i][1])
        i += 1
      if end == farthest:
        return -1
      ans += 1
      end = farthest

    return ans


# Link: https://leetcode.com/problems/minimum-penalty-for-a-shop/description/
class Solution:
  def bestClosingTime(self, customers: str) -> int:
    # Instead of computing the minimum penalty, we can compute the maximum profit.
    ans = 0
    profit = 0
    maxProfit = 0

    for i, customer in enumerate(customers):
      profit += 1 if customer == 'Y' else -1
      if profit > maxProfit:
        maxProfit = profit
        ans = i + 1

    return ans


# Link: https://leetcode.com/problems/maximum-product-after-k-increments/description/
class Solution:
  def maximumProduct(self, nums: List[int], k: int) -> int:
    kMod = 1_000_000_007
    ans = 1
    minHeap = nums.copy()
    heapq.heapify(minHeap)

    for _ in range(k):
      minNum = heapq.heappop(minHeap)
      heapq.heappush(minHeap, minNum + 1)

    while minHeap:
      ans *= heapq.heappop(minHeap)
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/minimum-index-of-a-valid-split/description/
class Solution:
  def minimumIndex(self, nums: List[int]) -> int:
    count1 = collections.Counter()
    count2 = collections.Counter(nums)

    for i, num in enumerate(nums):
      count1[num] = count1[num] + 1
      count2[num] = count2[num] - 1
      if count1[num] * 2 > i + 1 and count2[num] * 2 > len(nums) - i - 1:
        return i

    return -1


# Link: https://leetcode.com/problems/widest-pair-of-indices-with-equal-range-sum/description/
class Solution:
  def widestPairOfIndices(self, nums1: List[int], nums2: List[int]) -> int:
    ans = 0
    prefix = 0
    prefixToIndex = {0: -1}

    for i, (num1, num2) in enumerate(zip(nums1, nums2)):
      prefix += num1 - num2
      ans = max(ans, i - prefixToIndex.setdefault(prefix, i))

    return ans


# Link: https://leetcode.com/problems/mini-parser/description/
class Solution:
  def deserialize(self, s: str) -> NestedInteger:
    if s[0] != '[':
      return NestedInteger(int(s))

    stack = []

    for i, c in enumerate(s):
      if c == '[':
        stack.append(NestedInteger())
        start = i + 1
      elif c == ',':
        if i > start:
          num = int(s[start:i])
          stack[-1].add(NestedInteger(num))
        start = i + 1
      elif c == ']':
        popped = stack.pop()
        if i > start:
          num = int(s[start:i])
          popped.add(NestedInteger(num))
        if stack:
          stack[-1].add(popped)
        else:
          return popped
        start = i + 1


# Link: https://leetcode.com/problems/most-profitable-path-in-a-tree/description/
class Solution:
  def mostProfitablePath(self, edges: List[List[int]], bob: int, amount: List[int]) -> int:
    n = len(amount)
    tree = [[] for _ in range(n)]
    parent = [0] * n
    aliceDist = [-1] * n

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    # Fills `parent` and `aliceDist`.
    def dfs(u: int, prev: int, d: int) -> None:
      parent[u] = prev
      aliceDist[u] = d
      for v in tree[u]:
        if aliceDist[v] == -1:
          dfs(v, u, d + 1)

    dfs(0, -1, 0)

    # Modify amount athe path from node bob to node 0.
    # For each node,
    #   1. If Bob reaches earlier than Alice does, change the amount to 0.
    #   2. If Bob and Alice reach simultaneously, devide the amount by 2.
    u = bob
    bobDist = 0
    while u != 0:
      if bobDist < aliceDist[u]:
        amount[u] = 0
      elif bobDist == aliceDist[u]:
        amount[u] //= 2
      u = parent[u]
      bobDist += 1

    return self._getMoney(tree, 0, -1, amount)

  def _getMoney(self, tree: List[List[int]], u: int, prev: int, amount: List[int]) -> int:
    # a leaf node
    if len(tree[u]) == 1 and tree[u][0] == prev:
      return amount[u]

    maxPath = -math.inf
    for v in tree[u]:
      if v != prev:
        maxPath = max(maxPath, self._getMoney(tree, v, u, amount))

    return amount[u] + maxPath


# Link: https://leetcode.com/problems/find-the-celebrity/description/
# The knows API is already defined for you.
# Returns a bool, whether a knows b
# Def knows(a: int, b: int) -> bool:


class Solution:
  def findCelebrity(self, n: int) -> int:
    candidate = 0

    # Everyone knows the celebrity.
    for i in range(1, n):
      if knows(candidate, i):
        candidate = i

    # The candidate knows nobody and everyone knows the celebrity.
    for i in range(n):
      if i < candidate and knows(candidate, i) or not knows(i, candidate):
        return -1
      if i > candidate and not knows(i, candidate):
        return -1

    return candidate


# Link: https://leetcode.com/problems/different-ways-to-add-parentheses/description/
class Solution:
  @functools.lru_cache(None)
  def diffWaysToCompute(self, expression: str) -> List[int]:
    ans = []

    for i, c in enumerate(expression):
      if c in '+-*':
        for a in self.diffWaysToCompute(expression[:i]):
          for b in self.diffWaysToCompute(expression[i + 1:]):
            ans.append(eval(str(a) + c + str(b)))

    return ans or [int(expression)]


# Link: https://leetcode.com/problems/product-of-two-run-length-encoded-arrays/description/
class Solution:
  def findRLEArray(self, encoded1: List[List[int]],
                   encoded2: List[List[int]]) -> List[List[int]]:
    ans = []
    i = 0  # encoded1's index
    j = 0  # encoded2's index

    while i < len(encoded1) and j < len(encoded2):
      mult = encoded1[i][0] * encoded2[j][0]
      minFreq = min(encoded1[i][1], encoded2[j][1])
      if ans and mult == ans[-1][0]:
        ans[-1][1] += minFreq
      else:
        ans.append([mult, minFreq])
      encoded1[i][1] -= minFreq
      encoded2[j][1] -= minFreq
      if encoded1[i][1] == 0:
        i += 1
      if encoded2[j][1] == 0:
        j += 1

    return ans


# Link: https://leetcode.com/problems/is-graph-bipartite/description/
from enum import Enum


class Color(Enum):
  kWhite = 0
  kRed = 1
  kGreen = 2


class Solution:
  def isBipartite(self, graph: List[List[int]]) -> bool:
    colors = [Color.kWhite] * len(graph)

    for i in range(len(graph)):
      # This node has been colored, so do nothing.
      if colors[i] != Color.kWhite:
        continue
      # Always paint red for a white node.
      colors[i] = Color.kRed
      # BFS.
      q = collections.deque([i])
      while q:
        u = q.popleft()
        for v in graph[u]:
          if colors[v] == colors[u]:
            return False
          if colors[v] == Color.kWhite:
            colors[v] = Color.kRed if colors[u] == Color.kGreen else Color.kGreen
            q.append(v)

    return True


# Link: https://leetcode.com/problems/is-graph-bipartite/description/
from enum import Enum


class Color(Enum):
  kWhite = 0
  kRed = 1
  kGreen = 2


class Solution:
  def isBipartite(self, graph: List[List[int]]) -> bool:
    colors = [Color.kWhite] * len(graph)

    def isValidColor(u: int, color: Color) -> bool:
      # The painted color should be same as `color`.
      if colors[u] != Color.kWhite:
        return colors[u] == color

      colors[u] = color

      # All the children should have valid colors.
      childrenColor = Color.kRed if colors[u] == Color.kGreen else Color.kGreen
      return all(isValidColor(v, childrenColor) for v in graph[u])

    return all(colors[i] != Color.kWhite or isValidColor(i, Color.kRed)
               for i in range(len(graph)))


# Link: https://leetcode.com/problems/destroy-sequential-targets/description/
class Solution:
  def destroyTargets(self, nums: List[int], space: int) -> int:
    count = collections.Counter([num % space for num in nums])
    maxCount = max(count.values())
    return min(num for num in nums if count[num % space] == maxCount)


# Link: https://leetcode.com/problems/minimize-result-by-adding-parentheses-to-expression/description/
class Solution:
  def minimizeResult(self, expression: str) -> str:
    plusIndex = expression.index('+')
    left = expression[:plusIndex]
    right = expression[plusIndex + 1:]
    ans = ''
    mini = math.inf

    # the expression -> a * (b + c) * d
    for i in range(len(left)):
      for j in range(len(right)):
        a = 1 if i == 0 else int(left[:i])
        b = int(left[i:])
        c = int(right[0:j + 1])
        d = 1 if j == len(right) - 1 else int(right[j + 1:])
        val = a * (b + c) * d
        if val < mini:
          mini = val
          ans = ('' if i == 0 else str(a)) + \
              '(' + str(b) + '+' + str(c) + ')' + \
                ('' if j == len(right) - 1 else str(d))

    return ans


# Link: https://leetcode.com/problems/count-ways-to-build-good-strings/description/
class Solution:
  def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    # dp[i] := the number of good strings with length i
    dp = [1] + [0] * high

    for i in range(1, high + 1):
      if i >= zero:
        dp[i] = (dp[i] + dp[i - zero]) % kMod
      if i >= one:
        dp[i] = (dp[i] + dp[i - one]) % kMod
      if i >= low:
        ans = (ans + dp[i]) % kMod

    return ans


# Link: https://leetcode.com/problems/pour-water/description/
class Solution:
  def pourWater(self, heights: List[int], volume: int, k: int) -> List[int]:
    i = k

    while volume > 0:
      volume -= 1
      while i > 0 and heights[i] >= heights[i - 1]:
        i -= 1
      while i + 1 < len(heights) and heights[i] >= heights[i + 1]:
        i += 1
      while i > k and heights[i] == heights[i - 1]:
        i -= 1
      heights[i] += 1

    return heights


# Link: https://leetcode.com/problems/analyze-user-website-visit-pattern/description/
class Solution:
  def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    userToSites = collections.defaultdict(list)

    # Sort websites of each user by timestamp.
    for user, _, site in sorted(zip(username, timestamp, website), key=lambda x: (x[1])):
      userToSites[user].append(site)

    # For each of three websites, count its frequency.
    patternCount = collections.Counter()

    for user, sites in userToSites.items():
      patternCount.update(Counter(set(itertools.combinations(sites, 3))))

    return max(sorted(patternCount), key=patternCount.get)


# Link: https://leetcode.com/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/description/
class Solution:
  def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
    nameToMinutes = collections.defaultdict(list)

    for name, time in zip(keyName, keyTime):
      minutes = self._getMinutes(time)
      nameToMinutes[name].append(minutes)

    return sorted([name for name, minutes in nameToMinutes.items()
                   if self._hasAlert(minutes)])

  def _hasAlert(self, minutes: List[int]) -> bool:
    if len(minutes) > 70:
      return True
    minutes.sort()
    for i in range(2, len(minutes)):
      if minutes[i - 2] + 60 >= minutes[i]:
        return True
    return False

  def _getMinutes(self, time: str) -> int:
    h, m = map(int, time.split(':'))
    return 60 * h + m


# Link: https://leetcode.com/problems/wiggle-sort/description/
class Solution:
  def wiggleSort(self, nums: List[int]) -> None:
    # 1. If i is even, then nums[i] <= nums[i - 1].
    # 2. If i is odd, then nums[i] >= nums[i - 1].
    for i in range(1, len(nums)):
      if not (i & 1) and nums[i] > nums[i - 1] or \
              (i & 1) and nums[i] < nums[i - 1]:
        nums[i], nums[i - 1] = nums[i - 1], nums[i]


# Link: https://leetcode.com/problems/solve-the-equation/description/
class Solution:
  def solveEquation(self, equation: str) -> str:
    def calculate(s: str) -> tuple:
      coefficient = 0
      constant = 0
      num = 0
      sign = 1

      for i, c in enumerate(s):
        if c.isdigit():
          num = num * 10 + ord(c) - ord('0')
        elif c in '+-':
          constant += sign * num
          sign = 1 if c == '+' else -1
          num = 0
        else:
          if i > 0 and num == 0 and s[i - 1] == '0':
            continue
          coefficient += sign if num == 0 else sign * num
          num = 0

      return coefficient, constant + sign * num

    lhsEquation, rhsEquation = equation.split('=')
    lhsCoefficient, lhsConstant = calculate(lhsEquation)
    rhsCoefficient, rhsConstant = calculate(rhsEquation)
    coefficient = lhsCoefficient - rhsCoefficient
    constant = rhsConstant - lhsConstant

    if coefficient == 0 and constant == 0:
      return "Infinite solutions"
    if coefficient == 0 and constant != 0:
      return "No solution"
    return "x=" + str(constant // coefficient)


# Link: https://leetcode.com/problems/single-number-ii/description/
class Solution:
  def singleNumber(self, nums: List[int]) -> int:
    ones = 0
    twos = 0

    for num in nums:
      ones ^= (num & ~twos)
      twos ^= (num & ~ones)

    return ones


# Link: https://leetcode.com/problems/single-number-ii/description/
class Solution:
  def singleNumber(self, nums: List[int]) -> int:
    ones = 0
    twos = 0

    for num in nums:
      ones ^= num & ~twos
      twos ^= num & ~ones

    return ones


# Link: https://leetcode.com/problems/sort-linked-list-already-sorted-using-absolute-values/description/
class Solution:
  def sortLinkedList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = head
    curr = head.next

    while curr:
      if curr.val < 0:
        prev.next = curr.next
        curr.next = head
        head = curr
        curr = prev.next
      else:
        prev = curr
        curr = curr.next

    return head


# Link: https://leetcode.com/problems/collecting-chocolates/description/
class Solution:
  def minCost(self, nums: List[int], x: int) -> int:
    n = len(nums)
    ans = math.inf
    # minCost[i] := the minimum cost to collect the i-th type
    minCost = [math.inf] * n

    for rotate in range(n):
      for i in range(n):
        minCost[i] = min(minCost[i], nums[(i - rotate + n) % n])
      ans = min(ans, sum(minCost) + rotate * x)

    return ans


# Link: https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/description/
class Solution:
  def numSteps(self, s: str) -> int:
    ans = 0
    chars = [c for c in s]

    # All the trailing 0s can be popped by 1 step.
    while chars[-1] == '0':
      chars.pop()
      ans += 1

    if ''.join(chars) == '1':
      return ans

    # `s` is now odd, so add 1 to `s` and cost 1 step.
    # All the 1s will become 0s and can be popped by 1 step.
    # All the 0s will become 1s and can be popped by 2 steps (adding 1 then
    # dividing by 2).
    return ans + 1 + sum(1 if c == '1' else 2 for c in chars)


# Link: https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/description/
class Solution:
  def maxEvents(self, events: List[List[int]]) -> int:
    ans = 0
    minHeap = []
    i = 0  # events' index

    events.sort(key=lambda x: x[0])

    while minHeap or i < len(events):
      # If no events are available to attend today, let time flies to the next
      # available event.
      if not minHeap:
        d = events[i][0]
      # All the events starting from today are newly available.
      while i < len(events) and events[i][0] == d:
        heapq.heappush(minHeap, events[i][1])
        i += 1
      # Greedily attend the event that'll end the earliest since it has higher
      # chance can't be attended in the future.
      heapq.heappop(minHeap)
      ans += 1
      d += 1
      # Pop the events that can't be attended.
      while minHeap and minHeap[0] < d:
        heapq.heappop(minHeap)

    return ans


# Link: https://leetcode.com/problems/apply-operations-to-make-all-array-elements-equal-to-zero/description/
class Solution:
  def checkArray(self, nums: List[int], k: int) -> bool:
    if k == 1:
      return True

    needDecrease = 0
    # Store nums[i - k + 1..i] with decreasing nums[i - k + 1].
    dq = collections.deque()

    for i, num in enumerate(nums):
      if i >= k:
        needDecrease -= dq.popleft()
      if nums[i] < needDecrease:
        return False
      decreasedNum = nums[i] - needDecrease
      dq.append(decreasedNum)
      needDecrease += decreasedNum

    return dq[-1] == 0


# Link: https://leetcode.com/problems/maximum-sum-circular-subarray/description/
class Solution:
  def maxSubarraySumCircular(self, nums: List[int]) -> int:
    totalSum = 0
    currMaxSum = 0
    currMinSum = 0
    maxSum = -math.inf
    minSum = math.inf

    for num in nums:
      totalSum += num
      currMaxSum = max(currMaxSum + num, num)
      currMinSum = min(currMinSum + num, num)
      maxSum = max(maxSum, currMaxSum)
      minSum = min(minSum, currMinSum)

    return maxSum if maxSum < 0 else max(maxSum, totalSum - minSum)


# Link: https://leetcode.com/problems/maximum-profitable-triplets-with-increasing-prices-i/description/
class FenwickTree:
  def __init__(self, n: int):
    self.vals = [0] * (n + 1)

  def update(self, i: int, val: int) -> None:
    while i < len(self.vals):
      self.vals[i] = max(self.vals[i], val)
      i += FenwickTree.lowbit(i)

  def get(self, i: int) -> int:
    res = 0
    while i > 0:
      res = max(res, self.vals[i])
      i -= FenwickTree.lowbit(i)
    return res

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i


class Solution:
  def maxProfit(self, prices: List[int], profits: List[int]) -> int:
    ans = -1
    maxPrice = max(prices)
    maxProfitTree1 = FenwickTree(maxPrice)
    maxProfitTree2 = FenwickTree(maxPrice)

    for price, profit in zip(prices, profits):
      # max(proftis[i])
      maxProfit1 = maxProfitTree1.get(price - 1)
      # max(proftis[i]) + max(profits[j])
      maxProfit2 = maxProfitTree2.get(price - 1)
      maxProfitTree1.update(price, profit)
      if maxProfit1 > 0:
        maxProfitTree2.update(price, profit + maxProfit1)
      if maxProfit2 > 0:
        ans = max(ans, profit + maxProfit2)

    return ans


# Link: https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/description/
class Solution:
  def minimizeMax(self, nums: List[int], p: int) -> int:
    nums.sort()

    def numPairs(maxDiff: int) -> int:
      """
      Returns the number of pairs that can be obtained if the difference between
      each pair <= `maxDiff`.
      """
      pairs = 0
      i = 1
      while i < len(nums):
        # Greedily pair nums[i] with nums[i - 1].
        if nums[i] - nums[i - 1] <= maxDiff:
          pairs += 1
          i += 2
        else:
          i += 1
      return pairs

    return bisect.bisect_left(
        range(0, nums[-1] - nums[0]), p,
        key=lambda m: numPairs(m))


# Link: https://leetcode.com/problems/count-beautiful-substrings-i/description/
class Solution:
  def beautifulSubstrings(self, s: str, k: int) -> int:
    kVowels = 'aeiou'
    root = self._getRoot(k)
    ans = 0
    vowels = 0
    vowelsMinusConsonants = 0
    # {(vowels, vowelsMinusConsonants): count}
    prefixCount = collections.Counter({(0, 0): 1})

    for c in s:
      if c in kVowels:
        vowelsMinusConsonants += 1
        vowels = (vowels + 1) % root
      else:
        vowelsMinusConsonants -= 1
      ans += prefixCount[(vowels, vowelsMinusConsonants)]
      prefixCount[(vowels, vowelsMinusConsonants)] += 1

    return ans

  def _getRoot(self, k: int) -> int:
    for i in range(1, k + 1):
      if i * i % k == 0:
        return i


# Link: https://leetcode.com/problems/count-nodes-equal-to-sum-of-descendants/description/
from dataclasses import dataclass


@dataclass(frozen=True)
class T:
  summ: int
  count: int


class Solution:
  def equalToDescendants(self, root: Optional[TreeNode]) -> int:
    def dfs(root: Optional[TreeNode]) -> T:
      if not root:
        return T(0, 0)
      left = dfs(root.left)
      right = dfs(root.right)
      return T(root.val + left.summ + right.summ,
               left.count + right.count +
               (1 if root.val == left.summ + right.summ else 0))

    return dfs(root).count


# Link: https://leetcode.com/problems/minimum-speed-to-arrive-on-time/description/
class Solution:
  def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
    ans = -1
    l = 1
    r = int(1e7)

    def time(speed: int) -> float:
      summ = 0
      for i in range(len(dist) - 1):
        summ += math.ceil(dist[i] / speed)
      return summ + dist[-1] / speed

    while l <= r:
      m = (l + r) // 2
      if time(m) > hour:
        l = m + 1
      else:
        ans = m
        r = m - 1

    return ans


# Link: https://leetcode.com/problems/palindromic-substrings/description/
class Solution:
  def countSubstrings(self, s: str) -> int:
    def extendPalindromes(l: int, r: int) -> int:
      count = 0

      while l >= 0 and r < len(s) and s[l] == s[r]:
        count += 1
        l -= 1
        r += 1

      return count

    ans = 0

    for i in range(len(s)):
      ans += extendPalindromes(i, i)
      ans += extendPalindromes(i, i + 1)

    return ans


# Link: https://leetcode.com/problems/smallest-value-after-replacing-with-sum-of-prime-factors/description/
class Solution:
  def smallestValue(self, n: int) -> int:
    def getPrimeSum(n: int) -> int:
      primeSum = 0
      for i in range(2, n + 1):
        while n % i == 0:
          n //= i
          primeSum += i
      return primeSum

    primeSum = getPrimeSum(n)
    while n != primeSum:
      n = primeSum
      primeSum = getPrimeSum(n)
    return n


# Link: https://leetcode.com/problems/closest-fair-integer/description/
class Solution:
  def closestFair(self, n: int) -> int:
    digitsCount = len(str(n))
    return self._getOddDigits(digitsCount) if digitsCount & 1 else self._getEvenDigits(n)

  def _getOddDigits(self, digitsCount: int) -> int:
    zeros = (digitsCount + 1) // 2
    ones = (digitsCount - 1) // 2
    return int('1' + '0' * zeros + '1' * ones)

  def _getEvenDigits(self, n: int) -> int:
    digitsCount = len(str(n))
    maxNum = int('1' + '0' * digitsCount)
    for num in range(n, maxNum):
      if self._isValidNum(num):
        return num
    return self._getOddDigits(digitsCount + 1)

  def _isValidNum(self, num: int) -> bool:
    count = 0
    for c in str(num):
      count += -1 if ord(c) - ord('0') & 1 else 1
    return count == 0


# Link: https://leetcode.com/problems/largest-magic-square/description/
class Solution:
  def largestMagicSquare(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    # prefixRow[i][j] := the sum of the first j numbers in the i-th row
    prefixRow = [[0] * (n + 1) for _ in range(m)]
    # prefixCol[i][j] := the sum of the first j numbers in the i-th column
    prefixCol = [[0] * (m + 1) for _ in range(n)]

    for i in range(m):
      for j in range(n):
        prefixRow[i][j + 1] = prefixRow[i][j] + grid[i][j]
        prefixCol[j][i + 1] = prefixCol[j][i] + grid[i][j]

    def isMagicSquare(i: int, j: int, k: int) -> bool:
      """Returns True if grid[i..i + k)[j..j + k) is a magic square."""
      diag, antiDiag = 0, 0
      for d in range(k):
        diag += grid[i + d][j + d]
        antiDiag += grid[i + d][j + k - 1 - d]
      if diag != antiDiag:
        return False
      for d in range(k):
        if self._getSum(prefixRow, i + d, j, j + k - 1) != diag:
          return False
        if self._getSum(prefixCol, j + d, i, i + k - 1) != diag:
          return False
      return True

    def containsMagicSquare(k: int) -> bool:
      """Returns True if the grid contains any magic square of size k x k."""
      for i in range(m - k + 1):
        for j in range(n - k + 1):
          if isMagicSquare(i, j, k):
            return True
      return False

    for k in range(min(m, n), 1, -1):
      if containsMagicSquare(k):
        return k

    return 1

  def _getSum(self, prefix: List[List[int]], i: int, l: int, r: int) -> int:
    """Returns sum(grid[i][l..r]) or sum(grid[l..r][i])."""
    return prefix[i][r + 1] - prefix[i][l]


# Link: https://leetcode.com/problems/shortest-distance-to-target-color/description/
class Solution:
  def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
    kNumColor = 3
    n = len(colors)
    ans = []
    # left[i][c] := the closest index of color c in index i to the left
    left = [[0] * (kNumColor + 1) for _ in range(n)]
    # right[i][c] := the closest index of color c in index i to the right
    right = [[0] * (kNumColor + 1) for _ in range(n)]

    colorToLatestIndex = [0, -1, -1, -1]  # 0-indexed, -1 means N//A
    for i, color in enumerate(colors):
      colorToLatestIndex[color] = i
      for c in range(1, kNumColor + 1):
        left[i][c] = colorToLatestIndex[c]

    colorToLatestIndex = [0, -1, -1, -1]  # Reset.
    for i in range(n - 1, -1, -1):
      colorToLatestIndex[colors[i]] = i
      for c in range(1, kNumColor + 1):
        right[i][c] = colorToLatestIndex[c]

    for i, c in queries:
      leftDist = math.inf if left[i][c] == -1 else i - left[i][c]
      rightDist = math.inf if right[i][c] == -1 else right[i][c] - i
      minDist = min(leftDist, rightDist)
      ans.append(-1 if minDist == math.inf else minDist)

    return ans


# Link: https://leetcode.com/problems/seat-reservation-manager/description/
class SeatManager:
  def __init__(self, n: int):
    self.minHeap = [i + 1 for i in range(n)]

  def reserve(self) -> int:
    return heapq.heappop(self.minHeap)

  def unreserve(self, seatNumber: int) -> None:
    heapq.heappush(self.minHeap, seatNumber)


# Link: https://leetcode.com/problems/node-with-highest-edge-score/description/
class Solution:
  def edgeScore(self, edges: List[int]) -> int:
    scores = [0] * len(edges)
    for i, edge in enumerate(edges):
      scores[edge] += i
    return scores.index(max(scores))


# Link: https://leetcode.com/problems/k-th-smallest-prime-fraction/description/
class Solution:
  def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    n = len(arr)
    ans = [0, 1]
    l = 0
    r = 1

    while True:
      m = (l + r) / 2
      ans[0] = 0
      count = 0
      j = 1

      for i in range(n):
        while j < n and arr[i] > m * arr[j]:
          j += 1
        count += n - j
        if j == n:
          break
        if ans[0] * arr[j] < ans[1] * arr[i]:
          ans[0] = arr[i]
          ans[1] = arr[j]

      if count < k:
        l = m
      elif count > k:
        r = m
      else:
        return ans


# Link: https://leetcode.com/problems/decremental-string-concatenation/description/
class Solution:
  def minimizeConcatenatedLength(self, words: List[str]) -> int:
    @functools.lru_cache(None)
    def dp(i: int, first: str, last: str) -> int:
      """
      Returns the minimum concatenated length of the first i words starting with
      `first` and ending in `last`.
      """
      if i == len(words):
        return 0
      nextFirst = words[i][0]
      nextLast = words[i][-1]
      return len(words[i]) + min(
          # join(words[i - 1], words[i])
          dp(i + 1, first, nextLast) - (last == nextFirst),
          # join(words[i], words[i - 1])
          dp(i + 1, nextFirst, last) - (first == nextLast)
      )

    return len(words[0]) + dp(1, words[0][0], words[0][-1])


# Link: https://leetcode.com/problems/binary-searchable-numbers-in-an-unsorted-array/description/
class Solution:
  def binarySearchableNumbers(self, nums: List[int]) -> int:
    n = len(nums)
    # prefixMaxs[i] := max(nums[0..i))
    prefixMaxs = [0] * n
    # suffixMins[i] := min(nums[i + 1..n))
    suffixMins = [0] * n

    # Fill in `prefixMaxs`.
    prefixMaxs[0] = -math.inf
    for i in range(1, n):
      prefixMaxs[i] = max(prefixMaxs[i - 1], nums[i - 1])

    # Fill in `suffixMins`.
    suffixMins[n - 1] = math.inf
    for i in range(n - 2, -1, -1):
      suffixMins[i] = min(suffixMins[i + 1], nums[i + 1])

    return sum(prefixMaxs[i] < nums[i] < suffixMins[i] for i in range(n))


# Link: https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/
class Solution:
  def search(self, nums: List[int], target: int) -> bool:
    l = 0
    r = len(nums) - 1

    while l <= r:
      m = (l + r) // 2
      if nums[m] == target:
        return True
      if nums[l] == nums[m] == nums[r]:
        l += 1
        r -= 1
      elif nums[l] <= nums[m]:  # nums[l..m] are sorted
        if nums[l] <= target < nums[m]:
          r = m - 1
        else:
          l = m + 1
      else:  # nums[m..n - 1] are sorted
        if nums[m] < target <= nums[r]:
          l = m + 1
        else:
          r = m - 1

    return False


# Link: https://leetcode.com/problems/reachable-nodes-with-restrictions/description/
class Solution:
  def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
    tree = [[] for _ in range(n)]
    seen = set(restricted)

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int) -> int:
      if u in seen:
        return 0
      seen.add(u)
      return 1 + sum(dfs(v) for v in tree[u])

    return dfs(0)


# Link: https://leetcode.com/problems/valid-tic-tac-toe-state/description/
class Solution:
  def validTicTacToe(self, board: List[str]) -> bool:
    def isWin(c: str) -> bool:
      return any(row.count(c) == 3 for row in board) or \
          any(row.count(c) == 3 for row in list(zip(*board))) or \
          all(board[i][i] == c for i in range(3)) or \
          all(board[i][2 - i] == c for i in range(3))

    countX = sum(row.count('X') for row in board)
    countO = sum(row.count('O') for row in board)

    if countX < countO or countX - countO > 1:
      return False
    if isWin('X') and countX == countO or isWin('O') and countX != countO:
      return False

    return True


# Link: https://leetcode.com/problems/01-matrix/description/
class Solution:
  def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(mat)
    n = len(mat[0])
    q = collections.deque()

    for i in range(m):
      for j in range(n):
        if mat[i][j] == 0:
          q.append((i, j))
        else:
          mat[i][j] = math.inf

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if mat[x][y] <= mat[i][j] + 1:
          continue
        q.append((x, y))
        mat[x][y] = mat[i][j] + 1

    return mat


# Link: https://leetcode.com/problems/01-matrix/description/
class Solution:
  def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(mat)
    n = len(mat[0])
    q = collections.deque()
    seen = [[False] * n for _ in range(m)]

    for i in range(m):
      for j in range(n):
        if mat[i][j] == 0:
          q.append((i, j))
          seen[i][j] = True

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if seen[x][y]:
          continue
        mat[x][y] = mat[i][j] + 1
        q.append((x, y))
        seen[x][y] = True

    return mat


# Link: https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/description/
class Solution:
  def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def dfs(root: Optional[TreeNode]) -> Tuple[int, int]:
      nonlocal ans
      if not root:
        return (0, 0)
      leftSum, leftCount = dfs(root.left)
      rightSum, rightCount = dfs(root.right)
      summ = root.val + leftSum + rightSum
      count = 1 + leftCount + rightCount
      if summ // count == root.val:
        ans += 1
      return (summ, count)

    dfs(root)
    return ans


# Link: https://leetcode.com/problems/minimum-processing-time/description/
class Solution:
  def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
    return max(time + task
               for (time, task) in zip(sorted(processorTime), sorted(tasks)[::-4]))


# Link: https://leetcode.com/problems/count-lattice-points-inside-a-circle/description/
class Solution:
  def countLatticePoints(self, circles: List[List[int]]) -> int:
    return sum(any((xc - x)**2 + (yc - y)**2 <= r**2 for xc, yc, r in circles)
               for x in range(201)
               for y in range(201))


# Link: https://leetcode.com/problems/count-lattice-points-inside-a-circle/description/
class Solution:
  def countLatticePoints(self, circles: List[List[int]]) -> int:
    points = set()

    # dx := relative to x
    # dy := relative to y
    # So, dx^2 + dy^2 = r^2.
    for x, y, r in circles:
      for dx in range(-r, r + 1):
        yMax = int((r**2 - dx**2)**0.5)
        for dy in range(-yMax, yMax + 1):
          points.add((x + dx, y + dy))

    return len(points)


# Link: https://leetcode.com/problems/3sum-closest/description/
class Solution:
  def threeSumClosest(self, nums: List[int], target: int) -> int:
    ans = nums[0] + nums[1] + nums[2]

    nums.sort()

    for i in range(len(nums) - 2):
      if i > 0 and nums[i] == nums[i - 1]:
        continue
      # Choose nums[i] as the first number in the triplet, then search the
      # remaining numbers in [i + 1, n - 1].
      l = i + 1
      r = len(nums) - 1
      while l < r:
        summ = nums[i] + nums[l] + nums[r]
        if summ == target:
          return summ
        if abs(summ - target) < abs(ans - target):
          ans = summ
        if summ < target:
          l += 1
        else:
          r -= 1

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/description/
class Solution:
  def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
    # Greedily consider strings with `minSize`, so ignore `maxSize`.
    ans = 0
    letters = 0
    count = collections.Counter()
    substringCount = collections.Counter()

    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      if count[c] == 1:
        letters += 1
      while letters > maxLetters or r - l + 1 > minSize:
        count[s[l]] -= 1
        if count[s[l]] == 0:
          letters -= 1
        l += 1
      if r - l + 1 == minSize:
        sub = s[l:l + minSize]
        substringCount[sub] += 1
        ans = max(ans, substringCount[sub])

    return ans


# Link: https://leetcode.com/problems/sort-the-matrix-diagonally/description/
class Solution:
  def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
    m = len(mat)
    n = len(mat[0])

    count = collections.defaultdict(list)

    for i in range(m):
      for j in range(n):
        count[i - j].append(mat[i][j])

    for value in count.values():
      value.sort(reverse=1)

    for i in range(m):
      for j in range(n):
        mat[i][j] = count[i - j].pop()

    return mat


# Link: https://leetcode.com/problems/split-bst/description/
class Solution:
  def splitBST(self, root: Optional[TreeNode], target: int) -> List[Optional[TreeNode]]:
    if not root:
      return None, None
    if root.val > target:
      left, right = self.splitBST(root.left, target)
      root.left = right
      return left, root
    else:  # root.val <= target
      left, right = self.splitBST(root.right, target)
      root.right = left
      return root, right


# Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/
class Solution:
  def maxProfit(self, prices: List[int], fee: int) -> int:
    sell = 0
    hold = -math.inf

    for price in prices:
      sell = max(sell, hold + price)
      hold = max(hold, sell - price - fee)

    return sell


# Link: https://leetcode.com/problems/check-if-it-is-possible-to-split-array/description/
class Solution:
  def canSplitArray(self, nums: List[int], m: int) -> bool:
    return len(nums) < 3 or any(a + b >= m for a, b in itertools.pairwise(nums))


# Link: https://leetcode.com/problems/number-of-provinces/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1
    self.count -= 1

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def findCircleNum(self, isConnected: List[List[int]]) -> int:
    n = len(isConnected)
    uf = UnionFind(n)

    for i in range(n):
      for j in range(i, n):
        if isConnected[i][j] == 1:
          uf.unionByRank(i, j)

    return uf.count


# Link: https://leetcode.com/problems/search-in-rotated-sorted-array/description/
class Solution:
  def search(self, nums: List[int], target: int) -> int:
    l = 0
    r = len(nums) - 1

    while l <= r:
      m = (l + r) // 2
      if nums[m] == target:
        return m
      if nums[l] <= nums[m]:  # nums[l..m] are sorted.
        if nums[l] <= target < nums[m]:
          r = m - 1
        else:
          l = m + 1
      else:  # nums[m..n - 1] are sorted.
        if nums[m] < target <= nums[r]:
          l = m + 1
        else:
          r = m - 1

    return -1


# Link: https://leetcode.com/problems/strobogrammatic-number-ii/description/
class Solution:
  def findStrobogrammatic(self, n: int) -> List[str]:
    def helper(n: int, k: int) -> List[str]:
      if n == 0:
        return ['']
      if n == 1:
        return ['0', '1', '8']

      ans = []

      for inner in helper(n - 2, k):
        if n < k:
          ans.append('0' + inner + '0')
        ans.append('1' + inner + '1')
        ans.append('6' + inner + '9')
        ans.append('8' + inner + '8')
        ans.append('9' + inner + '6')

      return ans

    return helper(n, n)


# Link: https://leetcode.com/problems/partition-array-into-disjoint-intervals/description/
class Solution:
  def partitionDisjoint(self, nums: List[int]) -> int:
    n = len(nums)
    mini = [0] * (n - 1) + [nums[-1]]
    maxi = -math.inf

    for i in range(n - 2, - 1, -1):
      mini[i] = min(mini[i + 1], nums[i])

    for i, num in enumerate(nums):
      maxi = max(maxi, num)
      if maxi <= mini[i + 1]:
        return i + 1


# Link: https://leetcode.com/problems/find-the-k-th-lucky-number/description/
class Solution:
  def kthLuckyNumber(self, k: int) -> str:
    return bin(k + 1)[3:].replace('0', '4').replace('1', '7')


# Link: https://leetcode.com/problems/convert-bst-to-greater-tree/description/
class Solution:
  def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    prefix = 0

    def reversedInorder(root: Optional[TreeNode]) -> None:
      nonlocal prefix
      if not root:
        return

      reversedInorder(root.right)
      prefix += root.val
      root.val = prefix
      reversedInorder(root.left)

    reversedInorder(root)
    return root


# Link: https://leetcode.com/problems/find-the-most-competitive-subsequence/description/
class Solution:
  def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
    ans = []

    for i, num in enumerate(nums):
      # If |ans| - 1 + |nums[i..n)| >= k, then it means we still have enough
      # numbers, and we can safely pop an element from ans.
      while ans and ans[-1] > nums[i] and len(ans) - 1 + len(nums) - i >= k:
        ans.pop()
      if len(ans) < k:
        ans.append(nums[i])

    return ans


# Link: https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/description/
class Solution:
  def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    ans = 0
    prefix = 0
    prefixToIndex = {0: -1}

    for i, num in enumerate(nums):
      prefix += num
      target = prefix - k
      if target in prefixToIndex:
        ans = max(ans, i - prefixToIndex[target])
      if prefix not in prefixToIndex:
        prefixToIndex[prefix] = i

    return ans


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/description/
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
    nodes = set(nodes)

    def lca(root: 'TreeNode') -> 'TreeNode':
      if not root:
        return None
      if root in nodes:
        return root
      left = lca(root.left)
      right = lca(root.right)
      if left and right:
        return root
      return left or right

    return lca(root)


# Link: https://leetcode.com/problems/sum-of-beauty-of-all-substrings/description/
class Solution:
  def beautySum(self, s: str) -> int:
    ans = 0

    for i in range(len(s)):
      count = collections.Counter()
      for j in range(i, len(s)):
        count[s[j]] += 1
        ans += max(count.values()) - min(count.values())

    return ans


# Link: https://leetcode.com/problems/find-the-substring-with-maximum-cost/description/
class Solution:
  def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
    ans = 0
    cost = 0
    costs = [i for i in range(1, 27)]  # costs[i] := the cost of 'a' + i

    for c, val in zip(chars, vals):
      costs[ord(c) - ord('a')] = val

    for c in s:
      cost = max(0, cost + costs[ord(c) - ord('a')])
      ans = max(ans, cost)

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-lines-to-cover-points/description/
class Solution:
  def minimumLines(self, points: List[List[int]]) -> int:
    n = len(points)
    allCovered = (1 << n) - 1
    maxLines = n // 2 + (n & 1)

    def getSlope(p: List[int], q: List[int]) -> Tuple[int, int]:
      dx = p[0] - q[0]
      dy = p[1] - q[1]
      if dx == 0:
        return (0, p[0])
      if dy == 0:
        return (p[1], 0)
      d = gcd(dx, dy)
      x = dx // d
      y = dy // d
      return (x, y) if x > 0 else (-x, -y)

    @functools.lru_cache(None)
    def dfs(covered: int) -> int:
      if covered == allCovered:
        return 0

      ans = maxLines

      for i in range(n):
        if covered >> i & 1:
          continue
        for j in range(n):
          if i == j:
            continue
          # Connect the points[i] with the points[j].
          newCovered = covered | 1 << i | 1 << j
          slope = getSlope(points[i], points[j])
          # Mark the points covered by this line.
          for k in range(n):
            if getSlope(points[i], points[k]) == slope:
              newCovered |= 1 << k
          ans = min(ans, 1 + dfs(newCovered))

      return ans

    return dfs(0)


# Link: https://leetcode.com/problems/3sum-with-multiplicity/description/
class Solution:
  def threeSumMulti(self, arr: List[int], target: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    count = collections.Counter(arr)

    for i, x in count.items():
      for j, y in count.items():
        k = target - i - j
        if k not in count:
          continue
        if i == j and j == k:
          ans = (ans + x * (x - 1) * (x - 2) // 6) % kMod
        elif i == j and j != k:
          ans = (ans + x * (x - 1) // 2 * count[k]) % kMod
        elif i < j and j < k:
          ans = (ans + x * y * count[k]) % kMod

    return ans % kMod


# Link: https://leetcode.com/problems/minimum-number-of-food-buckets-to-feed-the-hamsters/description/
class Solution:
  def minimumBuckets(self, street: str) -> int:
    A = list(street)

    for i, c in enumerate(A):
      if c == 'H':
        if i > 0 and A[i - 1] == 'B':
          continue
        if i + 1 < len(A) and A[i + 1] == '.':
          # Always prefer place a bucket in (i + 1) because it enhances the
          # possibility to collect the upcoming houses.
          A[i + 1] = 'B'
        elif i > 0 and A[i - 1] == '.':
          A[i - 1] = 'B'
        else:
          return -1

    return A.count('B')


# Link: https://leetcode.com/problems/most-frequent-prime/description/
class Solution:
  def mostFrequentPrime(self, mat: List[List[int]]) -> int:
    dirs = ((1, 0), (1, -1), (0, -1), (-1, -1),
            (-1, 0), (-1, 1), (0, 1), (1, 1))
    m = len(mat)
    n = len(mat[0])
    count = collections.Counter()

    def isPrime(num: int) -> bool:
      return not any(num % i == 0 for i in range(2, int(num**0.5 + 1)))

    for i in range(m):
      for j in range(n):
        for dx, dy in dirs:
          num = 0
          x = i
          y = j
          while 0 <= x < m and 0 <= y < n:
            num = num * 10 + mat[x][y]
            if num > 10 and isPrime(num):
              count[num] += 1
            x += dx
            y += dy

    if not count.items():
      return -1
    return max(count.items(), key=lambda x: (x[1], x[0]))[0]


# Link: https://leetcode.com/problems/validate-stack-sequences/description/
class Solution:
  def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    stack = []
    i = 0  # popped's index

    for x in pushed:
      stack.append(x)
      while stack and stack[-1] == popped[i]:
        stack.pop()
        i += 1

    return not stack


# Link: https://leetcode.com/problems/number-of-black-blocks/description/
class Solution:
  def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
    ans = [0] * 5
    # count[i * n + j] := the number of black cells in
    # (i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)
    count = collections.Counter()

    for x, y in coordinates:
      for i in range(x, x + 2):
        for j in range(y, y + 2):
          # 2 x 2 submatrix with right-bottom conner being (i, j) contains the
          # current black cell (x, y).
          if 0 < i < m and 0 < j < n:
            count[(i, j)] += 1

    for freq in count.values():
      ans[freq] += 1

    ans[0] = (m - 1) * (n - 1) - sum(ans)
    return ans


# Link: https://leetcode.com/problems/find-the-minimum-possible-sum-of-a-beautiful-array/description/
class Solution:
  # Same as 2829. Determine the Minimum Sum of a k-avoiding Array
  def minimumPossibleSum(self, n: int, target: int) -> int:
    # These are the unique pairs that sum up to k (target):
    # (1, k - 1), (2, k - 2), ..., (ceil(k // 2), floor(k // 2)).
    # Our optimal strategy is to select 1, 2, ..., floor(k // 2), and then
    # choose k, k + 1, ... if necessary, as selecting any number in the range
    # [ceil(k // 2), k - 1] will result in a pair summing up to k.
    kMod = 1_000_000_007

    def trapezoid(a: int, b: int) -> int:
      """Returns sum(a..b)."""
      return (a + b) * (b - a + 1) // 2

    mid = target // 2  # floor(k // 2)
    if n <= mid:
      return trapezoid(1, n)
    return (trapezoid(1, mid) + trapezoid(target, target + (n - mid - 1))) % kMod


# Link: https://leetcode.com/problems/spiral-matrix-iii/description/
class Solution:
  def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    ans = [[rStart, cStart]]
    i = 0

    while len(ans) < rows * cols:
      for _ in range(i // 2 + 1):
        rStart += dy[i % 4]
        cStart += dx[i % 4]
        if 0 <= rStart < rows and 0 <= cStart < cols:
          ans.append([rStart, cStart])
      i += 1

    return ans


# Link: https://leetcode.com/problems/design-a-file-sharing-system/description/
from sortedcontainers import SortedSet


class FileSharing:
  def __init__(self, m: int):
    self.userToChunks: Dict[int, SortedSet[int]] = {}
    self.chunkToUsers: Dict[int, SortedSet[int]] = {}
    self.availableUserIds: List[int] = []

  def join(self, ownedChunks: List[int]) -> int:
    userId = heapq.heappop(self.availableUserIds) if self.availableUserIds \
      else len(self.userToChunks) + 1
    self.userToChunks[userId] = SortedSet(ownedChunks)
    for chunk in ownedChunks:
      self.chunkToUsers.setdefault(chunk, SortedSet()).add(userId)
    return userId

  def leave(self, userID: int) -> None:
    if userID not in self.userToChunks:
      return
    for chunk in self.userToChunks[userID]:
      self.chunkToUsers[chunk].discard(userID)
      if not self.chunkToUsers[chunk]:
        del self.chunkToUsers[chunk]
    del self.userToChunks[userID]
    heapq.heappush(self.availableUserIds, userID)

  def request(self, userID: int, chunkID: int) -> List[int]:
    if chunkID not in self.chunkToUsers:
      return []
    userIds = list(self.chunkToUsers[chunkID])
    self.userToChunks[userID].add(chunkID)
    self.chunkToUsers[chunkID].add(userID)
    return userIds


# Link: https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/description/
class Solution:
  def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
    ans = 0
    windowSum = 0

    for i in range(len(arr)):
      windowSum += arr[i]
      if i >= k:
        windowSum -= arr[i - k]
      if i >= k - 1 and windowSum // k >= threshold:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/maximum-consecutive-floors-without-special-floors/description/
class Solution:
  def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
    ans = 0

    special.sort()

    for a, b in zip(special, special[1:]):
      ans = max(ans, b - a - 1)

    return max(ans, special[0] - bottom, top - special[-1])


# Link: https://leetcode.com/problems/create-binary-tree-from-descriptions/description/
class Solution:
  def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
    children = set()
    valToNode = {}

    for p, c, isLeft in descriptions:
      parent = valToNode.setdefault(p, TreeNode(p))
      child = valToNode.setdefault(c, TreeNode(c))
      if isLeft:
        parent.left = child
      else:
        parent.right = child
      children.add(c)

    root = (set(valToNode) - set(children)).pop()
    return valToNode[root]


# Link: https://leetcode.com/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/description/
class Solution:
  def numOfPairs(self, nums: List[str], target: str) -> int:
    ans = 0
    count = collections.Counter()

    for num in nums:
      k = len(num)
      if target[:k] == num:
        ans += count[target[k:]]
      if target[-k:] == num:
        ans += count[target[:-k]]
      count[num] += 1

    return ans


# Link: https://leetcode.com/problems/longest-turbulent-subarray/description/
class Solution:
  def maxTurbulenceSize(self, arr: List[int]) -> int:
    ans = 1
    increasing = 1
    decreasing = 1

    for i in range(1, len(arr)):
      if arr[i] > arr[i - 1]:
        increasing = decreasing + 1
        decreasing = 1
      elif arr[i] < arr[i - 1]:
        decreasing = increasing + 1
        increasing = 1
      else:
        increasing = 1
        decreasing = 1
      ans = max(ans, max(increasing, decreasing))

    return ans


# Link: https://leetcode.com/problems/swap-nodes-in-pairs/description/
class Solution:
  def swapPairs(self, head: ListNode) -> ListNode:
    def getLength(head: ListNode) -> int:
      length = 0
      while head:
        length += 1
        head = head.next
      return length

    length = getLength(head)
    dummy = ListNode(0, head)
    prev = dummy
    curr = head

    for _ in range(length // 2):
      next = curr.next
      curr.next = next.next
      next.next = prev.next
      prev.next = next
      prev = curr
      curr = curr.next

    return dummy.next


# Link: https://leetcode.com/problems/generate-parentheses/description/
class Solution:
  def generateParenthesis(self, n):
    ans = []

    def dfs(l: int, r: int, s: str) -> None:
      if l == 0 and r == 0:
        ans.append(s)
      if l > 0:
        dfs(l - 1, r, s + '(')
      if l < r:
        dfs(l, r - 1, s + ')')

    dfs(n, n, '')
    return ans


# Link: https://leetcode.com/problems/longest-word-in-dictionary/description/
class Solution:
  def longestWord(self, words: List[str]) -> str:
    root = {}

    for word in words:
      node = root
      for c in word:
        if c not in node:
          node[c] = {}
        node = node[c]
      node['word'] = word

    def dfs(node: dict) -> str:
      ans = node['word'] if 'word' in node else ''

      for child in node:
        if 'word' in node[child] and len(node[child]['word']) > 0:
          childWord = dfs(node[child])
          if len(childWord) > len(ans) or (len(childWord) == len(ans) and childWord < ans):
            ans = childWord

      return ans

    return dfs(root)


# Link: https://leetcode.com/problems/k-divisible-elements-subarrays/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[int, TrieNode] = {}
    self.count = 0


class Solution:
  def countDistinct(self, nums: List[int], k: int, p: int) -> int:
    ans = 0
    root = TrieNode()

    def insert(node: TrieNode, i: int, k: int):
      nonlocal ans
      if i == len(nums) or k - (nums[i] % p == 0) < 0:
        return
      if nums[i] not in node.children:
        node.children[nums[i]] = TrieNode()
        ans += 1
      insert(node.children[nums[i]], i + 1, k - (nums[i] % p == 0))

    for i in range(len(nums)):
      insert(root, i, k)

    return ans


# Link: https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation/description/
class Solution:
  def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
    kMod = 1_000_000_007
    ans = 0
    # count[i] := the number of times nums[i] has been requested
    count = [0] * len(nums)

    for start, end in requests:
      count[start] += 1
      if end + 1 < len(nums):
        count[end + 1] -= 1

    for i in range(1, len(nums)):
      count[i] += count[i - 1]

    for num, c in zip(sorted(nums), sorted(count)):
      ans += num * c
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/coin-change-ii/description/
class Solution:
  def change(self, amount: int, coins: List[int]) -> int:
    dp = [1] + [0] * amount

    for coin in coins:
      for i in range(coin, amount + 1):
        dp[i] += dp[i - coin]

    return dp[amount]


# Link: https://leetcode.com/problems/implement-trie-prefix-tree/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.isWord = False


class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, word: str) -> None:
    node: TrieNode = self.root
    for c in word:
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True

  def search(self, word: str) -> bool:
    node: TrieNode = self._find(word)
    return node and node.isWord

  def startsWith(self, prefix: str) -> bool:
    return self._find(prefix)

  def _find(self, prefix: str) -> Optional[TrieNode]:
    node: TrieNode = self.root
    for c in prefix:
      if c not in node.children:
        return None
      node = node.children[c]
    return node


# Link: https://leetcode.com/problems/maximum-binary-tree-ii/description/
class Solution:
  def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if root.val < val:
      return TreeNode(val, root, None)
    curr = root
    while curr.right and curr.right.val > val:
      curr = curr.right
    inserted = TreeNode(val, curr.right, None)
    curr.right = inserted
    return root


# Link: https://leetcode.com/problems/maximum-binary-tree-ii/description/
class Solution:
  def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if not root:
      return TreeNode(val)
    if root.val < val:
      return TreeNode(val, root, None)
    root.right = self.insertIntoMaxTree(root.right, val)
    return root


# Link: https://leetcode.com/problems/reconstruct-a-2-row-binary-matrix/description/
class Solution:
  def reconstructMatrix(self, upper: int, lower: int, colsum: List[int]) -> List[List[int]]:
    if upper + lower != sum(colsum):
      return []
    if min(upper, lower) < colsum.count(2):
      return []

    ans = [[0] * len(colsum) for _ in range(2)]

    for j, c in enumerate(colsum):
      if c == 2:
        ans[0][j] = 1
        ans[1][j] = 1
        upper -= 1
        lower -= 1

    for j, c in enumerate(colsum):
      if c == 1 and upper > 0:
        ans[0][j] = 1
        c -= 1
        upper -= 1
      if c == 1 and lower > 0:
        ans[1][j] = 1
        lower -= 1

    return ans


# Link: https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/description/
class Solution:
  def maxScore(self, cardPoints: List[int], k: int) -> int:
    n = len(cardPoints)
    summ = sum(cardPoints)
    windowSum = sum(cardPoints[:n - k])
    ans = summ - windowSum

    for i in range(k):
      windowSum -= cardPoints[i]
      windowSum += cardPoints[i + n - k]
      ans = max(ans, summ - windowSum)

    return ans


# Link: https://leetcode.com/problems/one-edit-distance/description/
class Solution:
  def isOneEditDistance(self, s: str, t: str) -> bool:
    m = len(s)
    n = len(t)
    if m > n:  # Make sure that |s| <= |t|.
      return self.isOneEditDistance(t, s)

    for i in range(m):
      if s[i] != t[i]:
        if m == n:
          return s[i + 1:] == t[i + 1:]  # Replace s[i] with t[i].
        return s[i:] == t[i + 1:]  # Delete t[i].

    return m + 1 == n  # Delete t[-1].


# Link: https://leetcode.com/problems/minimum-total-space-wasted-with-k-resizing-operations/description/
class Solution:
  def minSpaceWastedKResizing(self, nums: List[int], k: int) -> int:
    kMax = 200_000_000

    @functools.lru_cache(None)
    def dp(i: int, k: int) -> int:
      """
      Returns the minimum space wasted for nums[i..n) if you can resize k times.
      """
      if i == len(nums):
        return 0
      if k == -1:
        return kMax

      res = kMax
      summ = 0
      maxNum = nums[i]

      for j in range(i, len(nums)):
        summ += nums[j]
        maxNum = max(maxNum, nums[j])
        wasted = maxNum * (j - i + 1) - summ
        res = min(res, dp(j + 1, k - 1) + wasted)

      return res

    return dp(0, k)


# Link: https://leetcode.com/problems/number-of-wonderful-substrings/description/
class Solution:
  def wonderfulSubstrings(self, word: str) -> int:
    ans = 0
    prefix = 0  # the binary prefix
    count = [0] * 1024  # the binary prefix count
    count[0] = 1  # the empty string ""

    for c in word:
      prefix ^= 1 << ord(c) - ord('a')
      # All the letters occur even number of times.
      ans += count[prefix]
      # `c` occurs odd number of times.
      ans += sum(count[prefix ^ 1 << i] for i in range(10))
      count[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/avoid-flood-in-the-city/description/
from sortedcontainers import SortedSet


class Solution:
  def avoidFlood(self, rains: List[int]) -> List[int]:
    ans = [-1] * len(rains)
    lakeIdToFullDay = {}
    emptyDays = SortedSet()  # indices of rains[i] == 0

    for i, lakeId in enumerate(rains):
      if lakeId == 0:
        emptyDays.add(i)
        continue
        # The lake was full in a previous day. Greedily find the closest day
        # to make the lake empty.
      if lakeId in lakeIdToFullDay:
        fullDay = lakeIdToFullDay[lakeId]
        emptyDayIndex = emptyDays.bisect_right(fullDay)
        if emptyDayIndex == len(emptyDays):  # Not found.
          return []
        # Empty the lake at this day.
        emptyDay = emptyDays[emptyDayIndex]
        ans[emptyDay] = lakeId
        emptyDays.discard(emptyDay)
      # The lake with `lakeId` becomes full at the day `i`.
      lakeIdToFullDay[lakeId] = i

    # Empty an arbitrary lake if there are remaining empty days.
    for emptyDay in emptyDays:
      ans[emptyDay] = 1

    return ans


# Link: https://leetcode.com/problems/meeting-rooms-ii/description/
class Solution:
  def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    n = len(intervals)
    ans = 0
    starts = []
    ends = []

    for start, end in intervals:
      starts.append(start)
      ends.append(end)

    starts.sort()
    ends.sort()

    j = 0
    for i in range(n):
      if starts[i] < ends[j]:
        ans += 1
      else:
        j += 1

    return ans


# Link: https://leetcode.com/problems/meeting-rooms-ii/description/
class Solution:
  def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    minHeap = []  # Store the end times of each room.

    for start, end in sorted(intervals):
      # There's no overlap, so we can reuse the same room.
      if minHeap and start >= minHeap[0]:
        heapq.heappop(minHeap)
      heapq.heappush(minHeap, end)

    return len(minHeap)


# Link: https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array/description/
class Solution:
  def minimumSwaps(self, nums: List[int]) -> int:
    minIndex = self._getLeftmostMinIndex(nums)
    maxIndex = self._getRightmostMaxIndex(nums)
    swaps = minIndex + (len(nums) - 1 - maxIndex)
    return swaps if minIndex <= maxIndex else swaps - 1

  def _getLeftmostMinIndex(self, nums: List[int]) -> int:
    min = nums[0]
    minIndex = 0
    for i in range(1, len(nums)):
      if nums[i] < min:
        min = nums[i]
        minIndex = i
    return minIndex

  def _getRightmostMaxIndex(self, nums: List[int]) -> int:
    max = nums[-1]
    maxIndex = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
      if nums[i] > max:
        max = nums[i]
        maxIndex = i
    return maxIndex


# Link: https://leetcode.com/problems/operations-on-tree/description/
class Node:
  def __init__(self):
    self.children: List[int] = []
    self.lockedBy = -1


class LockingTree:
  def __init__(self, parent: List[int]):
    self.parent = parent
    self.nodes = [Node() for _ in range(len(parent))]
    for i in range(1, len(parent)):
      self.nodes[parent[i]].children.append(i)

  def lock(self, num: int, user: int) -> bool:
    if self.nodes[num].lockedBy != -1:
      return False
    self.nodes[num].lockedBy = user
    return True

  def unlock(self, num: int, user: int) -> bool:
    if self.nodes[num].lockedBy != user:
      return False
    self.nodes[num].lockedBy = -1
    return True

  def upgrade(self, num: int, user: int) -> bool:
    if self.nodes[num].lockedBy != -1:
      return False
    if not self._anyLockedDescendant(num):
      return False

    # Walk up the hierarchy to ensure that there are no locked ancestors.
    i = num
    while i != -1:
      if self.nodes[i].lockedBy != -1:
        return False
      i = self.parent[i]

    self._unlockDescendants(num)
    self.nodes[num].lockedBy = user
    return True

  def _anyLockedDescendant(self, i: int) -> bool:
    return self.nodes[i].lockedBy != -1 or \
        any(self._anyLockedDescendant(child)
            for child in self.nodes[i].children)

  def _unlockDescendants(self, i: int) -> None:
    self.nodes[i].lockedBy = -1
    for child in self.nodes[i].children:
      self._unlockDescendants(child)


# Link: https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/description/
class Solution:
  # Similar to 253. Meeting Rooms II
  def minGroups(self, intervals: List[List[int]]) -> int:
    minHeap = []  # Stores `right`s.

    for left, right in sorted(intervals):
      if minHeap and left > minHeap[0]:
        # There is no overlaps, so we can reuse the same group.
        heapq.heappop(minHeap)
      heapq.heappush(minHeap, right)

    return len(minHeap)


# Link: https://leetcode.com/problems/minimum-operations-to-make-the-array-alternating/description/
class T:
  def __init__(self):
    self.count = collections.Counter()
    self.max = 0
    self.secondMax = 0
    self.maxFreq = 0
    self.secondMaxFreq = 0


class Solution:
  def minimumOperations(self, nums: List[int]) -> int:
    # 0 := odd indices, 1 := even indices
    ts = [T() for _ in range(2)]

    for i, num in enumerate(nums):
      t = ts[i & 1]
      t.count[num] += 1
      freq = t.count[num]
      if freq > t.maxFreq:
        t.maxFreq = freq
        t.max = num
      elif freq > t.secondMaxFreq:
        t.secondMaxFreq = freq
        t.secondMax = num

    if ts[0].max == ts[1].max:
      return len(nums) - max(ts[0].maxFreq + ts[1].secondMaxFreq,
                             ts[1].maxFreq + ts[0].secondMaxFreq)
    return len(nums) - (ts[0].maxFreq + ts[1].maxFreq)


# Link: https://leetcode.com/problems/number-of-single-divisor-triplets/description/
class Solution:
  def singleDivisorTriplet(self, nums: List[int]) -> int:
    ans = 0
    count = collections.Counter(nums)

    def divisible(summ: int, num: int) -> int:
      return summ % num == 0

    for a in range(1, 101):
      if count[a] == 0:
        continue
      for b in range(a, 101):
        if count[b] == 0:
          continue
        for c in range(b, 101):
          if count[c] == 0:
            continue
          summ = a + b + c
          if divisible(summ, a) + divisible(summ, b) + divisible(summ, c) != 1:
            continue
          if a == b:
            ans += count[a] * (count[a] - 1) // 2 * count[c]
          elif b == c:
            ans += count[b] * (count[b] - 1) // 2 * count[a]
          else:
            ans += count[a] * count[b] * count[c]

    return ans * 6


# Link: https://leetcode.com/problems/vowels-of-all-substrings/description/
class Solution:
  def countVowels(self, word: str) -> int:
    # dp[i] := the sum of the number of vowels of word[0..i), ...,
    # word[i - 1..i)
    dp = [0] * (len(word) + 1)

    for i, c in enumerate(word):
      dp[i + 1] = dp[i]
      if c in 'aeiou':
        dp[i + 1] += i + 1

    return sum(dp)


# Link: https://leetcode.com/problems/vowels-of-all-substrings/description/
class Solution:
  def countVowels(self, word: str) -> int:
    return sum((i + 1) * (len(word) - i)
               for i, c in enumerate(word)
               if c in 'aeiou')


# Link: https://leetcode.com/problems/length-of-the-longest-alphabetical-continuous-substring/description/
class Solution:
  def longestContinuousSubstring(self, s: str) -> int:
    ans = 1
    runningLen = 1

    for a, b in zip(s, s[1:]):
      if ord(a) + 1 == ord(b):
        runningLen += 1
        ans = max(ans, runningLen)
      else:
        runningLen = 1

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-x-and-y-equal/description/
class Solution:
  def minimumOperationsToMakeEqual(self, x, y):
    if x <= y:
      return y - x

    queue = collections.deque([x])
    seen = set()

    ans = 0
    while queue:
      for _ in range(len(queue)):
        num = queue.popleft()
        if num == y:
          return ans
        if num in seen:
          continue
        seen.add(num)
        if num % 11 == 0:
          queue.append(num // 11)
        if num % 5 == 0:
          queue.append(num // 5)
        queue.append(num - 1)
        queue.append(num + 1)
      ans += 1


# Link: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/
class Solution:
  def constructFromPrePost(self, pre: List[int], post: List[int]) -> Optional[TreeNode]:
    postToIndex = {num: i for i, num in enumerate(post)}

    def build(preStart: int, preEnd: int, postStart: int, postEnd: int) -> Optional[TreeNode]:
      if preStart > preEnd:
        return None
      if preStart == preEnd:
        return TreeNode(pre[preStart])

      rootVal = pre[preStart]
      leftRootVal = pre[preStart + 1]
      leftRootPostIndex = postToIndex[leftRootVal]
      leftSize = leftRootPostIndex - postStart + 1

      root = TreeNode(rootVal)
      root.left = build(preStart + 1, preStart + leftSize,
                        postStart, leftRootPostIndex)
      root.right = build(preStart + leftSize + 1, preEnd,
                         leftRootPostIndex + 1, postEnd - 1)
      return root

    return build(0, len(pre) - 1, 0, len(post) - 1)


# Link: https://leetcode.com/problems/bitwise-or-of-all-subsequence-sums/description/
class Solution:
  def subsequenceSumOr(self, nums: List[int]) -> int:
    ans = 0
    prefix = 0

    for num in nums:
      prefix += num
      ans |= num | prefix

    return ans


# Link: https://leetcode.com/problems/best-sightseeing-pair/description/
class Solution:
  def maxScoreSightseeingPair(self, values: List[int]) -> int:
    ans = 0
    bestPrev = 0

    for value in values:
      ans = max(ans, value + bestPrev)
      bestPrev = max(bestPrev, value) - 1

    return ans


# Link: https://leetcode.com/problems/count-ways-to-group-overlapping-ranges/description/
class Solution:
  def countWays(self, ranges: List[List[int]]) -> int:
    kMod = 1_000_000_007
    ans = 1
    prevEnd = -1

    for start, end in sorted(ranges):
      if start > prevEnd:
        ans = ans * 2 % kMod
      prevEnd = max(prevEnd, end)

    return ans


# Link: https://leetcode.com/problems/number-of-ways-to-select-buildings/description/
class Solution:
  def numberOfWays(self, s: str) -> int:
    ans = 0
    # before[i] := the number of i before the current digit
    before = [0] * 2
    # after[i] := the number of i after the current digit
    after = [0] * 2
    after[0] = s.count('0')
    after[1] = len(s) - after[0]

    for c in s:
      num = ord(c) - ord('0')
      after[num] -= 1
      if num == 0:
        ans += before[1] * after[1]
      else:
        ans += before[0] * after[0]
      before[num] += 1

    return ans


# Link: https://leetcode.com/problems/interleaving-string/description/
class Solution:
  def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    m = len(s1)
    n = len(s2)
    if m + n != len(s3):
      return False

    # dp[i][j] := true if s3[0..i + j) is formed by the interleaving of
    # s1[0..i) and s2[0..j)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
      dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]

    for j in range(1, n + 1):
      dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or \
            (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])

    return dp[m][n]


# Link: https://leetcode.com/problems/interleaving-string/description/
class Solution:
  def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    m = len(s1)
    n = len(s2)
    if m + n != len(s3):
      return False

    dp = [False] * (n + 1)

    for i in range(m + 1):
      for j in range(n + 1):
        if i == 0 and j == 0:
          dp[j] = True
        elif i == 0:
          dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
        elif j == 0:
          dp[j] = dp[j] and s1[i - 1] == s3[i - 1]
        else:
          dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or \
              (dp[j - 1] and s2[j - 1] == s3[i + j - 1])

    return dp[n]


# Link: https://leetcode.com/problems/make-string-a-subsequence-using-cyclic-increments/description/
class Solution:
  def canMakeSubsequence(self, str1: str, str2: str) -> bool:
    i = 0  # str2's index

    for c in str1:
      if c == str2[i] or chr(ord('a') + ((ord(c) - ord('a') + 1) % 26)) == str2[i]:
        i += 1
        if i == len(str2):
          return True

    return False


# Link: https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string/description/
class Solution:
  def minInsertions(self, s: str) -> int:
    neededRight = 0   # Increment by 2 for each '('.
    missingLeft = 0   # Increment by 1 for each missing '('.
    missingRight = 0  # Increment by 1 for each missing ')'.

    for c in s:
      if c == '(':
        if neededRight % 2 == 1:
          # e.g. '()(...'
          missingRight += 1
          neededRight -= 1
        neededRight += 2
      else:  # c == ')'
        neededRight -= 1
        if neededRight < 0:
          # e.g. '()))...'
          missingLeft += 1
          neededRight += 2

    return neededRight + missingLeft + missingRight


# Link: https://leetcode.com/problems/minimize-hamming-distance-after-swap-operations/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self.find(u)
    j = self.find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
    n = len(source)
    ans = 0
    uf = UnionFind(n)
    groupIdToCount = [collections.Counter() for _ in range(n)]

    for a, b in allowedSwaps:
      uf.unionByRank(a, b)

    for i in range(n):
      groupIdToCount[uf.find(i)][source[i]] += 1

    for i in range(n):
      groupId = uf.find(i)
      count = groupIdToCount[groupId]
      if target[i] not in count:
        ans += 1
      else:
        count[target[i]] -= 1
        if count[target[i]] == 0:
          del count[target[i]]

    return ans


# Link: https://leetcode.com/problems/maximum-score-after-applying-operations-on-a-tree/description/
class Solution:
  def maximumScoreAfterOperations(self, edges: List[List[int]], values: List[int]) -> int:
    tree = [[] for _ in values]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int, prev: int) -> None:
      if u > 0 and len(tree[u]) == 1:
        return values[u]
      childrenSum = sum(dfs(v, u)
                        for v in tree[u]
                        if v != prev)
      return min(childrenSum, values[u])

    return sum(values) - dfs(0, -1)


# Link: https://leetcode.com/problems/minimize-the-difference-between-target-and-chosen-elements/description/
class Solution:
  def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
    minSum = sum(min(row) for row in mat)
    if minSum >= target:  # No need to consider any larger combination.
      return minSum - target

    @functools.lru_cache(None)
    def dp(i: int, summ: int) -> int:
      if i == len(mat):
        return abs(summ - target)
      return min(dp(i + 1, summ + num) for num in mat[i])

    return dp(0, 0)


# Link: https://leetcode.com/problems/super-pow/description/
class Solution:
  def superPow(self, a: int, b: List[int]) -> int:
    kMod = 1337
    ans = 1

    for i in b:
      ans = pow(ans, 10, kMod) * pow(a, i, kMod)

    return ans % kMod


# Link: https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/description/
class Solution:
  def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
    if any(tile[1] - tile[0] + 1 >= carpetLen for tile in tiles):
      return carpetLen

    ans = 0
    prefix = [0] * (len(tiles) + 1)

    tiles.sort()
    starts = [tile[0] for tile in tiles]

    for i, tile in enumerate(tiles):
      length = tile[1] - tile[0] + 1
      prefix[i + 1] = prefix[i] + length

    for i, (s, _) in enumerate(tiles):
      carpetEnd = s + carpetLen - 1
      endIndex = bisect_right(starts, carpetEnd) - 1
      notCover = max(0, tiles[endIndex][1] - carpetEnd)
      ans = max(ans, prefix[endIndex + 1] - prefix[i] - notCover)

    return ans


# Link: https://leetcode.com/problems/complex-number-multiplication/description/
class Solution:
  def complexNumberMultiply(self, a: str, b: str) -> str:
    def getRealAndImag(s: str) -> tuple:
      return int(s[:s.index('+')]), int(s[s.index('+') + 1:-1])

    A, B = getRealAndImag(a)
    C, D = getRealAndImag(b)

    return str(A * C - B * D) + '+' + str(A * D + B * C) + 'i'


# Link: https://leetcode.com/problems/sum-of-beauty-in-the-array/description/
class Solution:
  def sumOfBeauties(self, nums: List[int]) -> int:
    n = len(nums)
    ans = 0
    minOfRight = [0] * (n - 1) + [nums[-1]]

    for i in range(n - 2, 1, -1):
      minOfRight[i] = min(nums[i], minOfRight[i + 1])

    maxOfLeft = nums[0]

    for i in range(1, n - 1):
      if maxOfLeft < nums[i] < minOfRight[i + 1]:
        ans += 2
      elif nums[i - 1] < nums[i] < nums[i + 1]:
        ans += 1
      maxOfLeft = max(maxOfLeft, nums[i])

    return ans


# Link: https://leetcode.com/problems/global-and-local-inversions/description/
class Solution:
  def isIdealPermutation(self, nums: List[int]) -> bool:
    for i, num in enumerate(nums):
      if abs(num - i) > 1:
        return False
    return True


# Link: https://leetcode.com/problems/global-and-local-inversions/description/
class Solution:
  def isIdealPermutation(self, nums: List[int]) -> bool:
    maxi = -1  # the number that is most likely > nums[i + 2]

    for i in range(len(nums) - 2):
      maxi = max(maxi, nums[i])
      if maxi > nums[i + 2]:
        return False

    return True


# Link: https://leetcode.com/problems/minimum-cost-to-reach-city-with-discounts/description/
class Solution:
  def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
    graph = [[] for _ in range(n)]
    minHeap = [(0, 0, discounts)]  # (d, u, leftDiscounts)
    minDiscounts = {}

    for city1, city2, toll in highways:
      graph[city1].append((city2, toll))
      graph[city2].append((city1, toll))

    while minHeap:
      d, u, leftDiscounts = heapq.heappop(minHeap)
      if u == n - 1:
        return d
      if u in minDiscounts and minDiscounts[u] >= leftDiscounts:
        continue
      minDiscounts[u] = leftDiscounts
      for v, w in graph[u]:
        heapq.heappush(minHeap, (d + w, v, leftDiscounts))
        if leftDiscounts > 0:
          heapq.heappush(minHeap, (d + w // 2, v, leftDiscounts - 1))

    return -1


# Link: https://leetcode.com/problems/meeting-scheduler/description/
class Solution:
  def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
    slots1.sort()
    slots2.sort()

    i = 0  # slots1's index
    j = 0  # slots2's index

    while i < len(slots1) and j < len(slots2):
      start = max(slots1[i][0], slots2[j][0])
      end = min(slots1[i][1], slots2[j][1])
      if start + duration <= end:
        return [start, start + duration]
      if slots1[i][1] < slots2[j][1]:
        i += 1
      else:
        j += 1

    return []


# Link: https://leetcode.com/problems/count-good-meals/description/
class Solution:
  def countPairs(self, deliciousness: List[int]) -> int:
    kMod = 10**9 + 7
    kMaxBit = 20 + 1
    ans = 0
    count = collections.Counter()

    for d in deliciousness:
      for i in range(kMaxBit + 1):
        power = 1 << i
        ans += count[power - d]
        ans %= kMod
      count[d] += 1

    return ans


# Link: https://leetcode.com/problems/maximum-strength-of-a-group/description/
class Solution:
  def maxStrength(self, nums: List[int]) -> int:
    posProd = 1
    negProd = 1
    maxNeg = -math.inf
    negCount = 0
    hasPos = False
    hasZero = False

    for num in nums:
      if num > 0:
        posProd *= num
        hasPos = True
      elif num < 0:
        negProd *= num
        maxNeg = max(maxNeg, num)
        negCount += 1
      else:  # num == 0
        hasZero = True

    if negCount == 0 and not hasPos:
      return 0
    if negCount % 2 == 0:
      return negProd * posProd
    if negCount >= 3:
      return negProd // maxNeg * posProd
    if hasPos:
      return posProd
    if hasZero:
      return 0
    return maxNeg


# Link: https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/description/
class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    maxNum = max(nums)
    ans = 0
    count = 0

    l = 0
    for r, num in enumerate(nums):
      if num == maxNum:
        count += 1
      # Keep the window to include k - 1 times of the maxNummum number.
      while count == k:
        if nums[l] == maxNum:
          count -= 1
        l += 1
      # If l > 0, nums[l:r+1] has k - 1 times of the maxNummum number. For any
      # subarray nums[i:r+1], where i < l, it will have at least k times of the
      # maxNummum number, since nums[l - 1] equals the maxNummum number.
      ans += l

    return ans


# Link: https://leetcode.com/problems/minimum-time-difference/description/
class Solution:
  def findMinDifference(self, timePoints: List[str]) -> int:
    ans = 24 * 60
    nums = sorted([int(timePoint[:2]) * 60 + int(timePoint[3:])
                   for timePoint in timePoints])

    for a, b in zip(nums, nums[1:]):
      ans = min(ans, b - a)

    return min(ans, 24 * 60 - nums[-1] + nums[0])


# Link: https://leetcode.com/problems/minimum-non-zero-product-of-the-array-elements/description/
class Solution:
  def minNonZeroProduct(self, p: int) -> int:
    kMod = 1_000_000_007
    # Can always turn [1..2^p - 1] to [1, 1, ..., 2^p - 2, 2^p - 2, 2^p - 1].
    n = 1 << p
    halfCount = n // 2 - 1
    return pow(n - 2, halfCount, kMod) * ((n - 1) % kMod) % kMod


# Link: https://leetcode.com/problems/number-of-boomerangs/description/
class Solution:
  def numberOfBoomerangs(self, points: List[List[int]]) -> int:
    ans = 0

    for x1, y1 in points:
      count = collections.Counter()
      for x2, y2 in points:
        ans += 2 * count[(x1 - x2)**2 + (y1 - y2)**2]
        count[(x1 - x2)**2 + (y1 - y2)**2] += 1

    return ans


# Link: https://leetcode.com/problems/bold-words-in-string/description/
class Solution:
  def boldWords(self, words: List[str], s: str) -> str:
    n = len(s)
    ans = []
    # bold[i] := True if s[i] should be bolded
    bold = [0] * n

    boldEnd = -1  # s[i:boldEnd] should be bolded
    for i in range(n):
      for word in words:
        if s[i:].startswith(word):
          boldEnd = max(boldEnd, i + len(word))
      bold[i] = boldEnd > i

    # Construct the string with the bold tags.
    i = 0
    while i < n:
      if bold[i]:
        j = i
        while j < n and bold[j]:
          j += 1
        # s[i..j) should be bolded.
        ans.append('<b>' + s[i:j] + '</b>')
        i = j
      else:
        ans.append(s[i])
        i += 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/bold-words-in-string/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.isWord = False


class Solution:
  def boldWords(self, words: List[str], s: str) -> str:
    n = len(s)
    ans = []
    # bold[i] := True if s[i] should be bolded
    bold = [0] * n
    root = TrieNode()

    def insert(word: str) -> None:
      node = root
      for c in word:
        if c not in node.children:
          node.children[c] = TrieNode()
        node = node.children[c]
      node.isWord = True

    def find(s: str, i: int) -> int:
      node = root
      ans = -1
      for j in range(i, len(s)):
        node = node.children.setdefault(s[j], TrieNode())
        if node.isWord:
          ans = j
      return ans

    for word in words:
      insert(word)

    boldEnd = -1  # `s[i..boldEnd]` should be bolded.
    for i in range(n):
      boldEnd = max(boldEnd, find(s, i))
      bold[i] = boldEnd >= i

    # Construct the with bold tags
    i = 0
    while i < n:
      if bold[i]:
        j = i
        while j < n and bold[j]:
          j += 1
        # `s[i..j)` should be bolded.
        ans.append('<b>' + s[i:j] + '</b>')
        i = j
      else:
        ans.append(s[i])
        i += 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/description/
class T:
  def __init__(self, leftMax: int, rightMax: int, subtreeMax: int):
    self.leftMax = leftMax
    self.rightMax = rightMax
    self.subtreeMax = subtreeMax


class Solution:
  def longestZigZag(self, root: Optional[TreeNode]) -> int:
    def dfs(root: Optional[TreeNode]) -> T:
      if not root:
        return T(-1, -1, -1)
      left = dfs(root.left)
      right = dfs(root.right)
      leftZigZag = left.rightMax + 1
      rightZigZag = right.leftMax + 1
      subtreeMax = max(leftZigZag, rightZigZag,
                       left.subtreeMax, right.subtreeMax)
      return T(leftZigZag, rightZigZag, subtreeMax)

    return dfs(root).subtreeMax


# Link: https://leetcode.com/problems/find-k-length-substrings-with-no-repeated-characters/description/
class Solution:
  def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
    ans = 0
    unique = 0
    count = collections.Counter()

    for i, c in enumerate(s):
      count[c] += 1
      if count[c] == 1:
        unique += 1
      if i >= k:
        count[s[i - k]] -= 1
        if count[s[i - k]] == 0:
          unique -= 1
        if unique == k:
          ans += 1

    return ans


# Link: https://leetcode.com/problems/sort-the-jumbled-numbers/description/
class Solution:
  def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
    def getMapped(num: int) -> int:
      mapped = []
      for c in str(num):
        mapped.append(str(mapping[ord(c) - ord('0')]))
      return int(''.join(mapped))
    A = [(getMapped(num), i, num) for i, num in enumerate(nums)]
    return [num for _, i, num in sorted(A)]


# Link: https://leetcode.com/problems/make-costs-of-paths-equal-in-a-binary-tree/description/
class Solution:
  def minIncrements(self, n: int, cost: List[int]) -> int:
    ans = 0

    for i in range(n // 2 - 1, -1, -1):
      l = i * 2 + 1
      r = i * 2 + 2
      ans += abs(cost[l] - cost[r])
      # Record the information in the parent from the children. So, there's need to actually
      # update the values in the children.
      cost[i] += max(cost[l], cost[r])

    return ans


# Link: https://leetcode.com/problems/loud-and-rich/description/
class Solution:
  def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
    graph = [[] for _ in range(len(quiet))]

    for v, u in richer:
      graph[u].append(v)

    @functools.lru_cache(None)
    def dfs(u: int) -> int:
      ans = u

      for v in graph[u]:
        res = dfs(v)
        if quiet[res] < quiet[ans]:
          ans = res

      return ans

    return map(dfs, range(len(graph)))


# Link: https://leetcode.com/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/description/
class Solution:
  def findMaximumNumber(self, k: int, x: int) -> int:
    def getSumPrices(num: int) -> int:
      """Returns the sum of prices of all numbers from 1 to `num`."""
      sumPrices = 0
      # Increment `num` to account the 0-th row in the count of groups.
      num += 1
      for i in range(num.bit_length(), 0, -1):
        if i % x == 0:
          groupSize = 1 << i
          halfGroupSize = 1 << i - 1
          sumPrices += num // groupSize * halfGroupSize
          sumPrices += max(0, (num % groupSize) - halfGroupSize)
      return sumPrices

    return bisect.bisect_right(range(1, 10**15), k,
                               key=lambda m: getSumPrices(m))


# Link: https://leetcode.com/problems/word-break/description/
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wordSet = set(wordDict)

    @functools.lru_cache(None)
    def wordBreak(s: str) -> bool:
      if s in wordSet:
        return True
      return any(s[:i] in wordSet and wordBreak(s[i:]) for i in range(len(s)))

    return wordBreak(s)


# Link: https://leetcode.com/problems/word-break/description/
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    n = len(s)
    maxLength = len(max(wordDict, key=len))
    wordSet = set(wordDict)
    # dp[i] := True if s[0..i) can be segmented
    dp = [True] + [False] * n

    for i in range(1, n + 1):
      for j in reversed(range(i)):
        if i - j > maxLength:
          break
        # s[0..j) can be segmented and s[j..i) is in the wordSet, so s[0..i)
        # can be segmented.
        if dp[j] and s[j:i] in wordSet:
          dp[i] = True
          break

    return dp[n]


# Link: https://leetcode.com/problems/word-break/description/
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wordSet = set(wordDict)

    @functools.lru_cache(None)
    def wordBreak(i: int) -> bool:
      """Returns True if s[i..n) can be segmented."""
      if i == len(s):
        return True
      return any(s[i:j] in wordSet and wordBreak(j) for j in range(i + 1, len(s) + 1))

    return wordBreak(0)


# Link: https://leetcode.com/problems/word-break/description/
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    n = len(s)
    wordSet = set(wordDict)
    # dp[i] := True if s[0..i) can be segmented
    dp = [True] + [False] * n

    for i in range(1, n + 1):
      for j in range(i):
        # s[0..j) can be segmented and s[j..i) is in `wordSet`, so s[0..i) can
        # be segmented.
        if dp[j] and s[j:i] in wordSet:
          dp[i] = True
          break

    return dp[n]


# Link: https://leetcode.com/problems/smallest-integer-divisible-by-k/description/
class Solution:
  def smallestRepunitDivByK(self, k: int) -> int:
    if k % 10 not in {1, 3, 7, 9}:
      return -1

    seen = set()
    n = 0

    for length in range(1, k + 1):
      n = (n * 10 + 1) % k
      if n == 0:
        return length
      if n in seen:
        return -1
      seen.add(n)

    return -1


# Link: https://leetcode.com/problems/equal-row-and-column-pairs/description/
class Solution:
  def equalPairs(self, grid: List[List[int]]) -> int:
    n = len(grid)
    ans = 0

    for i in range(n):
      for j in range(n):
        k = 0
        while k < n:
          if grid[i][k] != grid[k][j]:
            break
          k += 1
        if k == n:  # R[i] == C[j]
          ans += 1

    return ans


# Link: https://leetcode.com/problems/ugly-number-ii/description/
class Solution:
  def nthUglyNumber(self, n: int) -> int:
    nums = [1]
    i2 = 0
    i3 = 0
    i5 = 0

    while len(nums) < n:
      next2 = nums[i2] * 2
      next3 = nums[i3] * 3
      next5 = nums[i5] * 5
      next = min(next2, next3, next5)
      if next == next2:
        i2 += 1
      if next == next3:
        i3 += 1
      if next == next5:
        i5 += 1
      nums.append(next)

    return nums[-1]


# Link: https://leetcode.com/problems/integer-break/description/
class Solution:
  def integerBreak(self, n: int) -> int:
    # If an optimal product contains a factor f >= 4, then we can replace it
    # with 2 and f - 2 without losing optimality. As 2(f - 2) = 2f - 4 >= f,
    # we never need a factor >= 4, meaning we only need factors 1, 2, and 3
    # (and 1 is wasteful).
    # Also, 3 * 3 is better than 2 * 2 * 2, so we never use 2 more than twice.
    if n == 2:  # 1 * 1
      return 1
    if n == 3:  # 1 * 2
      return 2

    ans = 1

    while n > 4:
      n -= 3
      ans *= 3
    ans *= n

    return ans


# Link: https://leetcode.com/problems/soup-servings/description/
class Solution:
  def soupServings(self, n: int) -> float:
    @functools.lru_cache(None)
    def dfs(a: int, b: int) -> float:
      if a <= 0 and b <= 0:
        return 0.5
      if a <= 0:
        return 1.0
      if b <= 0:
        return 0.0
      return 0.25 * (dfs(a - 4, b) +
                     dfs(a - 3, b - 1) +
                     dfs(a - 2, b - 2) +
                     dfs(a - 1, b - 3))

    return 1 if n >= 4800 else dfs((n + 24) // 25, (n + 24) // 25)


# Link: https://leetcode.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/description/
class Solution:
  def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:
    prefix = [0] + list(itertools.accumulate(candiesCount))
    return [prefix[t] // c <= d < prefix[t + 1] for t, d, c in queries]


# Link: https://leetcode.com/problems/minimum-number-of-groups-to-create-a-valid-assignment/description/
class Solution:
  def minGroupsForValidAssignment(self, nums: List[int]) -> int:
    count = collections.Counter(nums)
    minFreq = min(count.values())

    for groupSize in range(minFreq, 0, -1):
      numGroups = self.getNumGroups(count, groupSize)
      if numGroups > 0:
        return numGroups

    raise ValueError("Invalid argument")

  def getNumGroups(self, count: Dict[int, int], groupSize: int) -> int:
    """Returns the number of groups if each group's size is `groupSize` or `groupSize + 1`."""
    numGroups = 0
    for freq in count.values():
      a = freq // (groupSize + 1)
      b = freq % (groupSize + 1)
      if b == 0:
        # Assign 1 number from `groupSize - b` out of `a` groups to this group,
        # so we'll have `a - (groupSize - b)` groups of size `groupSize + 1`
        # and `groupSize - b + 1` groups of size `groupSize`. In total, we have
        # `a + 1` groups.
        numGroups += a
      elif groupSize - b <= a:
        numGroups += a + 1
      else:
        return 0
    return numGroups


# Link: https://leetcode.com/problems/sliding-subarray-beauty/description/
class Solution:
  def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
    ans = []
    count = [0] * 50  # count[i] := the frequency of (i + 50)

    for i, num in enumerate(nums):
      if num < 0:
        count[num + 50] += 1
      if i - k >= 0 and nums[i - k] < 0:
        count[nums[i - k] + 50] -= 1
      if i + 1 >= k:
        ans.append(self._getXthSmallestNum(count, x))

    return ans

  def _getXthSmallestNum(self, count: List[int], x: int) -> int:
    prefix = 0
    for i in range(50):
      prefix += count[i]
      if prefix >= x:
        return i - 50
    return 0


# Link: https://leetcode.com/problems/minimum-number-of-people-to-teach/description/
class Solution:
  def minimumTeachings(self, n: int, languages: List[List[int]], friendships: List[List[int]]) -> int:
    languageSets = [set(languages) for languages in languages]
    needTeach = set()
    languageCount = collections.Counter()

    # Find friends that can't communicate.
    for u, v in friendships:
      if not languageSets[u - 1] & languageSets[v - 1]:
        needTeach.add(u - 1)
        needTeach.add(v - 1)

    # Find the most popular language.
    for u in needTeach:
      for language in languageSets[u]:
        languageCount[language] += 1

    # Teach the most popular language to people don't understand.
    return len(needTeach) - max(languageCount.values(), default=0)


# Link: https://leetcode.com/problems/partition-string-into-minimum-beautiful-substrings/description/
class Solution:
  def minimumBeautifulSubstrings(self, s: str) -> int:
    n = len(s)
    # dp[i] := the minimum number of beautiful substrings for the first i chars
    dp = [0] + [n + 1] * n

    for i in range(1, n + 1):
      if s[i - 1] == '0':
        continue
      num = 0  # the number of s[i - 1..j - 1]
      for j in range(i, n + 1):
        num = (num << 1) + int(s[j - 1])
        if self._isPowerOfFive(num):
          dp[j] = min(dp[j], dp[i - 1] + 1)

    return -1 if dp[n] == n + 1 else dp[n]

  def _isPowerOfFive(self, num: int) -> bool:
    while num % 5 == 0:
      num //= 5
    return num == 1


# Link: https://leetcode.com/problems/minimum-falling-path-sum/description/
class Solution:
  def minFallingPathSum(self, A: List[List[int]]) -> int:
    n = len(A)

    for i in range(1, n):
      for j in range(n):
        mini = math.inf
        for k in range(max(0, j - 1), min(n, j + 2)):
          mini = min(mini, A[i - 1][k])
        A[i][j] += mini

    return min(A[-1])


# Link: https://leetcode.com/problems/online-election/description/
class TopVotedCandidate:
  def __init__(self, persons: List[int], times: List[int]):
    self.times = times
    self.timeToLead = {}
    count = collections.Counter()  # {person: voted}
    lead = -1

    for person, time in zip(persons, times):
      count[person] += 1
      if count[person] >= count[lead]:
        lead = person
      self.timeToLead[time] = lead

  def q(self, t: int) -> int:
    i = bisect_right(self.times, t)
    return self.timeToLead[self.times[i - 1]]


# Link: https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/description/
class Solution:
  def maximumSubarraySum(self, nums: List[int], k: int) -> int:
    ans = 0
    summ = 0
    distinct = 0
    count = collections.Counter()

    for i, num in enumerate(nums):
      summ += num
      count[num] += 1
      if count[num] == 1:
        distinct += 1
      if i >= k:
        count[nums[i - k]] -= 1
        if count[nums[i - k]] == 0:
          distinct -= 1
        summ -= nums[i - k]
      if i >= k - 1 and distinct == k:
        ans = max(ans, summ)

    return ans


# Link: https://leetcode.com/problems/max-difference-you-can-get-from-changing-an-integer/description/
class Solution:
  def maxDiff(self, num: int) -> int:
    s = str(num)

    def firstNot(s: str, t: str) -> int:
      for i, c in enumerate(s):
        if all(c != d for d in t):
          return i
      return 0

    firstNot9 = firstNot(s, '9')
    firstNot01 = firstNot(s, '01')
    a = s.replace(s[firstNot9], '9')
    b = s.replace(s[firstNot01], '1' if firstNot01 == 0 else '0')
    return int(a) - int(b)


# Link: https://leetcode.com/problems/h-index-ii/description/
class Solution:
  def hIndex(self, citations: List[int]) -> int:
    n = len(citations)
    return n - bisect.bisect_left(range(n), n,
                                  key=lambda m: citations[m] + m)


# Link: https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/description/
class Solution:
  def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
    ans = []
    supplies = set(supplies)
    graph = collections.defaultdict(list)
    inDegrees = collections.Counter()
    q = collections.deque()

    # Build the graph.
    for i, recipe in enumerate(recipes):
      for ingredient in ingredients[i]:
        if ingredient not in supplies:
          graph[ingredient].append(recipe)
          inDegrees[recipe] += 1

    # Perform topological sorting.
    for recipe in recipes:
      if inDegrees[recipe] == 0:
        q.append(recipe)

    while q:
      u = q.popleft()
      ans.append(u)
      for v in graph[u]:
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return ans


# Link: https://leetcode.com/problems/2-keys-keyboard/description/
class Solution:
  def minSteps(self, n: int) -> int:
    if n <= 1:
      return 0

    # dp[i] := the minimum steps to get i 'A's
    # Copy 'A', then paste 'A' i - 1 times.
    dp = [i for i in range(n + 1)]

    for i in range(2, n + 1):
      for j in range(i // 2, 2, -1):
        if i % j == 0:
          dp[i] = dp[j] + i // j  # Paste dp[j] i / j times.
          break

    return dp[n]


# Link: https://leetcode.com/problems/find-pattern-in-infinite-stream-i/description/
# Definition for an infinite stream.
# class InfiniteStream:
#   def next(self) -> int:
#     pass

class Solution:
  def findPattern(self, stream: Optional['InfiniteStream'], pattern: List[int]) -> int:
    lps = self._getLPS(pattern)
    i = 0  # stream's index
    j = 0  # pattern's index
    bit = 0  # the bit in the stream
    readNext = False
    while True:
      if not readNext:
        bit = stream.next()
        readNext = True
      if bit == pattern[j]:
        i += 1
        readNext = False
        j += 1
        if j == len(pattern):
          return i - j
      # Mismatch after j matches.
      elif j > 0:
        # Don't match lps[0..lps[j - 1]] since they will match anyway.
        j = lps[j - 1]
      else:
        i += 1
        readNext = False

  def _getLPS(self, pattern: List[int]) -> List[int]:
    """
    Returns the lps array, where lps[i] is the length of the longest prefix of
    pattern[0..i] which is also a suffix of this substring.
    """
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
      while j > 0 and pattern[j] != pattern[i]:
        j = lps[j - 1]
      if pattern[i] == pattern[j]:
        j += 1
        lps[i] = j
    return lps


# Link: https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/description/
class Solution:
  def numRollsToTarget(self, n: int, k: int, target: int) -> int:
    kMod = 1_000_000_007
    dp = [1] + [0] * target

    for _ in range(n):  # n dices
      newDp = [0] * (target + 1)
      for i in range(1, k + 1):  # numbers 1, 2, ..., f
        for t in range(i, target + 1):  # all the possible targets
          newDp[t] += dp[t - i]
          newDp[t] %= kMod
      dp = newDp

    return dp[target]


# Link: https://leetcode.com/problems/first-unique-number/description/
class FirstUnique:
  def __init__(self, nums: List[int]):
    self.seen = set()
    self.unique = {}
    for num in nums:
      self.add(num)

  def showFirstUnique(self) -> int:
    return next(iter(self.unique), -1)

  def add(self, value: int) -> None:
    if value not in self.seen:
      self.seen.add(value)
      self.unique[value] = 1
    elif value in self.unique:
      # We have added this value before, and this is the second time we're
      # adding it. So, erase the value from `unique`.
      self.unique.pop(value)


# Link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
class Solution:
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    def countNodes(root: Optional[TreeNode]) -> int:
      if not root:
        return 0
      return 1 + countNodes(root.left) + countNodes(root.right)

    leftCount = countNodes(root.left)

    if leftCount == k - 1:
      return root.val
    if leftCount >= k:
      return self.kthSmallest(root.left, k)
    return self.kthSmallest(root.right, k - 1 - leftCount)  # leftCount < k


# Link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
class Solution:
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    rank = 0
    ans = 0

    def traverse(root: Optional[TreeNode]) -> None:
      nonlocal rank
      nonlocal ans
      if not root:
        return

      traverse(root.left)
      rank += 1
      if rank == k:
        ans = root.val
        return
      traverse(root.right)

    traverse(root)
    return ans


# Link: https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
class Solution:
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    stack = []

    while root:
      stack.append(root)
      root = root.left

    for _ in range(k - 1):
      root = stack.pop()
      root = root.right
      while root:
        stack.append(root)
        root = root.left

    return stack[-1].val


# Link: https://leetcode.com/problems/repeated-string-match/description/
class Solution:
  def repeatedStringMatch(self, a: str, b: str) -> int:
    n = math.ceil(len(b) / len(a))
    s = a * n
    if b in s:
      return n
    if b in s + a:
      return n + 1
    return -1


# Link: https://leetcode.com/problems/maximum-width-ramp/description/
class Solution:
  def maxWidthRamp(self, nums: List[int]) -> int:
    ans = 0
    stack = []

    for i, num in enumerate(nums):
      if stack == [] or num <= nums[stack[-1]]:
        stack.append(i)

    for i, num in reversed(list(enumerate(nums))):
      while stack and num >= nums[stack[-1]]:
        ans = max(ans, i - stack.pop())

    return ans


# Link: https://leetcode.com/problems/maximum-good-subarray-sum/description/
class Solution:
  def maximumSubarraySum(self, nums: List[int], k: int) -> int:
    ans = -math.inf
    prefix = 0
    numToMinPrefix = {}

    for num in nums:
      if num not in numToMinPrefix or numToMinPrefix[num] > prefix:
        numToMinPrefix[num] = prefix
      prefix += num
      if num + k in numToMinPrefix:
        ans = max(ans, prefix - numToMinPrefix[num + k])
      if num - k in numToMinPrefix:
        ans = max(ans, prefix - numToMinPrefix[num - k])

    return 0 if ans == -math.inf else ans


# Link: https://leetcode.com/problems/put-boxes-into-the-warehouse-ii/description/
class Solution:
  def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
    l = 0
    r = len(warehouse) - 1

    for box in sorted(boxes, reverse=True):
      if l > r:
        return len(warehouse)
      if box <= warehouse[l]:
        l += 1
      elif box <= warehouse[r]:
        r -= 1

    return l + (len(warehouse) - r - 1)


# Link: https://leetcode.com/problems/minimum-swaps-to-make-strings-equal/description/
class Solution:
  def minimumSwap(self, s1: str, s2: str) -> int:
    # ('xx', 'yy') = (2 'xy's) . 1 swap
    # ('yy', 'xx') = (2 'yx's) . 1 swap
    # ('xy', 'yx') = (1 'xy' and 1 'yx') . 2 swaps
    xy = 0  # the number of indices i's s.t. s1[i] = 'x' and s2[i] 'y'
    yx = 0  # the number of indices i's s.t. s1[i] = 'y' and s2[i] 'x'

    for a, b in zip(s1, s2):
      if a == b:
        continue
      if a == 'x':
        xy += 1
      else:
        yx += 1

    if (xy + yx) % 2 == 1:
      return -1
    return xy // 2 + yx // 2 + (2 if xy % 2 == 1 else 0)


# Link: https://leetcode.com/problems/remove-adjacent-almost-equal-characters/description/
class Solution:
  def removeAlmostEqualCharacters(self, word: str) -> int:
    ans = 0
    i = 1
    while i < len(word):
      if abs(ord(word[i]) - ord(word[i - 1])) <= 1:
        ans += 1
        i += 2
      else:
        i += 1
    return ans


# Link: https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/description/
class Solution:
  def removeSubfolders(self, folder: List[str]) -> List[str]:
    ans = []
    prev = ""

    folder.sort()

    for f in folder:
      if len(prev) > 0 and f.startswith(prev) and f[len(prev)] == '/':
        continue
      ans.append(f)
      prev = f

    return ans


# Link: https://leetcode.com/problems/sort-integers-by-the-power-value/description/
class Solution:
  def getKth(self, lo: int, hi: int, k: int) -> int:
    return sorted([(self._getPow(i), i) for i in range(lo, hi + 1)])[k - 1][1]

  def _getPow(self, n: int) -> int:
    if n == 1:
      return 0
    if n % 2 == 0:
      return 1 + self._getPow(n // 2)
    return 1 + self._getPow(n * 3 + 1)


# Link: https://leetcode.com/problems/maximum-square-area-by-removing-fences-from-a-field/description/
class Solution:
  def maximizeSquareArea(self, m: int, n: int, hFences: List[int], vFences: List[int]) -> int:
    hFences = sorted(hFences + [1, m])
    vFences = sorted(vFences + [1, n])
    hGaps = {hFences[i] - hFences[j]
             for i in range(len(hFences))
             for j in range(i)}
    vGaps = {vFences[i] - vFences[j]
             for i in range(len(vFences))
             for j in range(i)}
    maxGap = next((hGap
                  for hGap in sorted(hGaps, reverse=True)
                  if hGap in vGaps), -1)
    return -1 if maxGap == -1 else maxGap**2 % (10**9 + 7)


# Link: https://leetcode.com/problems/continuous-subarrays/description/
class Solution:
  def continuousSubarrays(self, nums: List[int]) -> int:
    ans = 1  # [nums[0]]
    left = nums[0] - 2
    right = nums[0] + 2
    l = 0

    # nums[l..r] is a valid window.
    for r in range(1, len(nums)):
      if left <= nums[r] <= right:
        left = max(left, nums[r] - 2)
        right = min(right, nums[r] + 2)
      else:
        # nums[r] is out-of-bounds, so reconstruct the window.
        left = nums[r] - 2
        right = nums[r] + 2
        l = r
        # If we consistently move leftward in each iteration, it implies that
        # the entire left subarray satisfies the given condition. For every
        # subarray with l in the range [0, r], the condition is met, preventing
        # the code from reaching the final "else" condition. Instead, it stops
        # at the "if" condition.
        while nums[r] - 2 <= nums[l] <= nums[r] + 2:
          left = max(left, nums[l] - 2)
          right = min(right, nums[l] + 2)
          l -= 1
        l += 1
      # nums[l..r], num[l + 1..r], ..., nums[r]
      ans += r - l + 1

    return ans


# Link: https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/description/
class Solution:
  def verifyPreorder(self, preorder: List[int]) -> List[int]:
    low = -math.inf
    stack = []

    for p in preorder:
      if p < low:
        return False
      while stack and stack[-1] < p:
        low = stack.pop()
      stack.append(p)

    return True


# Link: https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/description/
class Solution:
  def verifyPreorder(self, preorder: List[int]) -> bool:
    i = 0

    def dfs(min: int, max: int) -> None:
      nonlocal i
      if i == len(preorder):
        return
      if preorder[i] < min or preorder[i] > max:
        return

      val = preorder[i]
      i += 1
      dfs(min, val)
      dfs(val, max)

    dfs(-math.inf, math.inf)
    return i == len(preorder)


# Link: https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/description/
class Solution:
  def verifyPreorder(self, preorder: List[int]) -> bool:
    low = -math.inf
    i = -1

    for p in preorder:
      if p < low:
        return False
      while i >= 0 and preorder[i] < p:
        low = preorder[i]
        i -= 1
      i += 1
      preorder[i] = p

    return True


# Link: https://leetcode.com/problems/find-bottom-left-tree-value/description/
class Solution:
  def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
    q = collections.deque([root])

    while q:
      root = q.popleft()
      if root.right:
        q.append(root.right)
      if root.left:
        q.append(root.left)

    return root.val


# Link: https://leetcode.com/problems/find-bottom-left-tree-value/description/
class Solution:
  def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
    ans = 0
    maxDepth = 0

    def dfs(root: Optional[TreeNode], depth: int) -> None:
      nonlocal ans
      nonlocal maxDepth
      if not root:
        return
      if depth > maxDepth:
        maxDepth = depth
        ans = root.val

      dfs(root.left, depth + 1)
      dfs(root.right, depth + 1)

    dfs(root, 1)
    return ans


# Link: https://leetcode.com/problems/restore-the-array-from-adjacent-pairs/description/
class Solution:
  def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
    ans = []
    numToAdjs = collections.defaultdict(list)

    for a, b in adjacentPairs:
      numToAdjs[a].append(b)
      numToAdjs[b].append(a)

    for num, adjs in numToAdjs.items():
      if len(adjs) == 1:
        ans.append(num)
        ans.append(adjs[0])
        break

    while len(ans) < len(adjacentPairs) + 1:
      tail = ans[-1]
      prev = ans[-2]
      adjs = numToAdjs[tail]
      if adjs[0] == prev:  # adjs[0] is already used
        ans.append(adjs[1])
      else:
        ans.append(adjs[0])

    return ans


# Link: https://leetcode.com/problems/count-number-of-rectangles-containing-each-point/description/
class Solution:
  def countRectangles(self, rectangles: List[List[int]], points: List[List[int]]) -> List[int]:
    ans = []
    yToXs = [[] for _ in range(101)]

    for l, h in rectangles:
      yToXs[h].append(l)

    for xs in yToXs:
      xs.sort()

    for xi, yi in points:
      count = 0
      for y in range(yi, 101):
        xs = yToXs[y]
        count += len(xs) - bisect.bisect_left(xs, xi)
      ans.append(count)

    return ans


# Link: https://leetcode.com/problems/score-after-flipping-matrix/description/
class Solution:
  def matrixScore(self, grid: List[List[int]]) -> int:
    # Flip the rows with a leading 0.
    for row in grid:
      if row[0] == 0:
        self._flip(row)

    # Flip the columns with 1s < 0s.
    for j, col in enumerate(list(zip(*grid))):
      if sum(col) * 2 < len(grid):
        self._flipCol(grid, j)

    # Add a binary number for each row.
    return sum(self._binary(row) for row in grid)

  def _flip(self, row: List[int]) -> None:
    for i in range(len(row)):
      row[i] ^= 1

  def _flipCol(self, grid: List[List[int]], j: int) -> None:
    for i in range(len(grid)):
      grid[i][j] ^= 1

  def _binary(self, row: List[int]) -> int:
    res = row[0]
    for j in range(1, len(row)):
      res = res * 2 + row[j]
    return res


# Link: https://leetcode.com/problems/score-after-flipping-matrix/description/
class Solution:
  def matrixScore(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    ans = m  # All the cells in the first column are 1.

    for j in range(1, n):
      # The best strategy is flipping the rows with a leading 0..
      onesCount = sum(grid[i][j] == grid[i][0] for i in range(m))
      ans = ans * 2 + max(onesCount, m - onesCount)

    return ans


# Link: https://leetcode.com/problems/combination-sum-ii/description/
class Solution:
  def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    ans = []

    def dfs(s: int, target: int, path: List[int]) -> None:
      if target < 0:
        return
      if target == 0:
        ans.append(path.copy())
        return

      for i in range(s, len(candidates)):
        if i > s and candidates[i] == candidates[i - 1]:
          continue
        path.append(candidates[i])
        dfs(i + 1, target - candidates[i], path)
        path.pop()

    candidates.sort()
    dfs(0, target, [])
    return ans


# Link: https://leetcode.com/problems/construct-product-matrix/description/
class Solution:
  def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
    kMod = 12345
    m = len(grid)
    n = len(grid[0])
    ans = [[0] * n for _ in range(m)]
    prefix = [1]
    suffix = 1

    for row in grid:
      for cell in row:
        prefix.append(prefix[-1] * cell % kMod)

    for i in reversed(range(m)):
      for j in reversed(range(n)):
        ans[i][j] = prefix[i * n + j] * suffix % kMod
        suffix = suffix * grid[i][j] % kMod

    return ans


# Link: https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/description/
class Solution:
  def numSubseq(self, nums: List[int], target: int) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    ans = 0

    nums.sort()

    l = 0
    r = n - 1
    while l <= r:
      if nums[l] + nums[r] <= target:
        ans += pow(2, r - l, kMod)
        l += 1
      else:
        r -= 1

    return ans % kMod


# Link: https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/description/
class Solution:
  def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    def dfs(left: Optional[TreeNode], right: Optional[TreeNode], isOddLevel: bool) -> None:
      if not left:
        return
      if isOddLevel:
        left.val, right.val = right.val, left.val
      dfs(left.left, right.right, not isOddLevel)
      dfs(left.right, right.left, not isOddLevel)

    dfs(root.left, root.right, True)
    return root


# Link: https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/description/
class Solution:
  def smallestSubsequence(self, text: str) -> str:
    ans = []
    count = collections.Counter(text)
    used = [False] * 26

    for c in text:
      count[c] -= 1
      if used[ord(c) - ord('a')]:
        continue
      while ans and ans[-1] > c and count[ans[-1]] > 0:
        used[ord(ans[-1]) - ord('a')] = False
        ans.pop()
      ans.append(c)
      used[ord(ans[-1]) - ord('a')] = True

    return ''.join(ans)


# Link: https://leetcode.com/problems/nth-digit/description/
class Solution:
  def findNthDigit(self, n: int) -> int:
    def getDigit(num: int, pos: int, digitSize: int):
      if pos == 0:
        return num % 10
      for _ in range(digitSize - pos):
        num //= 10
      return num % 10

    digitSize = 1
    startNum = 1
    count = 9

    while digitSize * count < n:
      n -= digitSize * count
      digitSize += 1
      startNum *= 10
      count *= 10

    targetNum = startNum + (n - 1) // digitSize
    pos = n % digitSize

    return getDigit(targetNum, pos, digitSize)


# Link: https://leetcode.com/problems/number-of-distinct-substrings-in-a-string/description/
class Solution:
  def countDistinct(self, s: str) -> int:
    kBase = 26
    kMod = 1_000_000_007

    n = len(s)
    ans = 0
    pow = [1] + [0] * n     # pow[i] := kBase^i
    hashes = [0] * (n + 1)  # hashes[i] := the hash of s[0..i)

    def val(c: str) -> int:
      return ord(c) - ord('a')

    for i in range(1, n + 1):
      pow[i] = pow[i - 1] * kBase % kMod
      hashes[i] = (hashes[i - 1] * kBase + val(s[i - 1])) % kMod

    def getHash(l: int, r: int) -> int:
      """Returns the hash of s[l..r)."""
      hash = (hashes[r] - hashes[l] * pow[r - l]) % kMod
      return hash + kMod if hash < 0 else hash

    for length in range(1, n + 1):
      seen = set()
      for i in range(n - length + 1):
        seen.add(getHash(i, i + length))
      ans += len(seen)

    return ans


# Link: https://leetcode.com/problems/total-cost-to-hire-k-workers/description/
class Solution:
  def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
    ans = 0
    i = 0
    j = len(costs) - 1
    minHeapL = []  # First half
    minHeapR = []  # Second half

    for _ in range(k):
      while len(minHeapL) < candidates and i <= j:
        heapq.heappush(minHeapL, costs[i])
        i += 1
      while len(minHeapR) < candidates and i <= j:
        heapq.heappush(minHeapR, costs[j])
        j -= 1
      if not minHeapL:
        ans += heapq.heappop(minHeapR)
      elif not minHeapR:
        ans += heapq.heappop(minHeapL)
      # Both `minHeapL` and `minHeapR` are not empty.
      elif minHeapL[0] <= minHeapR[0]:
        ans += heapq.heappop(minHeapL)
      else:
        ans += heapq.heappop(minHeapR)

    return ans


# Link: https://leetcode.com/problems/minimum-time-to-revert-word-to-initial-state-i/description/
class Solution:
  # Same as 3029. Minimum Time to Revert Word to Initial State I
  def minimumTimeToInitialState(self, word: str, k: int) -> int:
    n = len(word)
    maxOps = (n - 1) // k + 1
    z = self._zFunction(word)

    for ans in range(1, maxOps):
      if z[ans * k] >= n - ans * k:
        return ans

    return maxOps

  def _zFunction(self, s: str) -> List[int]:
    """
    Returns the z array, where z[i] is the length of the longest prefix of
    s[i..n) which is also a prefix of s.

    https://cp-algorithms.com/string/z-function.html#implementation
    """
    n = len(s)
    z = [0] * n
    l = 0
    r = 0
    for i in range(1, n):
      if i < r:
        z[i] = min(r - i, z[i - l])
      while i + z[i] < n and s[z[i]] == s[i + z[i]]:
        z[i] += 1
      if i + z[i] > r:
        l = i
        r = i + z[i]
    return z


# Link: https://leetcode.com/problems/print-foobar-alternately/description/
from threading import Semaphore


class FooBar:
  def __init__(self, n):
    self.n = n
    self.fooSemaphore = Semaphore(1)
    self.barSemaphore = Semaphore(0)

  def foo(self, printFoo: 'Callable[[], None]') -> None:
    for _ in range(self.n):
      self.fooSemaphore.acquire()
      printFoo()
      self.barSemaphore.release()

  def bar(self, printBar: 'Callable[[], None]') -> None:
    for _ in range(self.n):
      self.barSemaphore.acquire()
      printBar()
      self.fooSemaphore.release()


# Link: https://leetcode.com/problems/subarray-sum-equals-k/description/
class Solution:
  def subarraySum(self, nums: List[int], k: int) -> int:
    ans = 0
    prefix = 0
    count = collections.Counter({0: 1})

    for num in nums:
      prefix += num
      ans += count[prefix - k]
      count[prefix] += 1

    return ans


# Link: https://leetcode.com/problems/minimum-factorization/description/
class Solution:
  def smallestFactorization(self, num: int) -> int:
    if num == 1:
      return 1

    ans = 0
    base = 1

    for i in range(9, 1, -1):
      while num % i == 0:
        num //= i
        ans = base * i + ans
        base *= 10

    return ans if num == 1 and ans < 2**31 - 1 else 0


# Link: https://leetcode.com/problems/maximize-the-profit-as-the-salesman/description/
class Solution:
  def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
    # dp[i] := the maximum amount of gold of selling the first i houses
    dp = [0] * (n + 1)
    endToStartAndGolds = [[] for _ in range(n)]

    for start, end, gold in offers:
      endToStartAndGolds[end].append((start, gold))

    for end in range(1, n + 1):
      # Get at least the same gold as selling the first `end - 1` houses.
      dp[end] = dp[end - 1]
      for start, gold in endToStartAndGolds[end - 1]:
        dp[end] = max(dp[end], dp[start] + gold)

    return dp[n]


# Link: https://leetcode.com/problems/longest-string-chain/description/
class Solution:
  def longestStrChain(self, words: List[str]) -> int:
    wordsSet = set(words)

    @functools.lru_cache(None)
    def dp(s: str) -> int:
      """Returns the longest chain where s is the last word."""
      ans = 1
      for i in range(len(s)):
        pred = s[:i] + s[i + 1:]
        if pred in wordsSet:
          ans = max(ans, dp(pred) + 1)
      return ans

    return max(dp(word) for word in words)


# Link: https://leetcode.com/problems/longest-string-chain/description/
class Solution:
  def longestStrChain(self, words: List[str]) -> int:
    dp = {}

    for word in sorted(words, key=len):
      dp[word] = max(dp.get(word[:i] + word[i + 1:], 0) +
                     1 for i in range(len(word)))

    return max(dp.values())


# Link: https://leetcode.com/problems/bitwise-xor-of-all-pairings/description/
class Solution:
  def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
    xors1 = functools.reduce(operator.xor, nums1)
    xors2 = functools.reduce(operator.xor, nums2)
    # If the size of nums1 is m and the size of nums2 is n, then each number in
    # nums1 is repeated n times and each number in nums2 is repeated m times.
    return (len(nums1) % 2 * xors2) ^ (len(nums2) % 2 * xors1)


# Link: https://leetcode.com/problems/divide-players-into-teams-of-equal-skill/description/
class Solution:
  def dividePlayers(self, skill: List[int]) -> int:
    n = len(skill)
    teamSkill = sum(skill) // (n // 2)
    ans = 0
    count = collections.Counter(skill)

    for s, freq in count.items():
      requiredSkill = teamSkill - s
      if count[requiredSkill] != freq:
        return -1
      ans += s * requiredSkill * freq

    return ans // 2


# Link: https://leetcode.com/problems/maximum-rows-covered-by-columns/description/
class Solution:
  def maximumRows(self, matrix: List[List[int]], numSelect: int) -> int:
    ans = 0

    def dfs(colIndex: int, leftColsCount: int, mask: int):
      nonlocal ans
      if leftColsCount == 0:
        ans = max(ans, self._getAllZerosRowCount(matrix, mask))
        return

      if colIndex == len(matrix[0]):
        return

      # Choose this column.
      dfs(colIndex + 1, leftColsCount - 1, mask | 1 << colIndex)
      # Don't choose this column.
      dfs(colIndex + 1, leftColsCount, mask)

    dfs(0, numSelect, 0)
    return ans

  def _getAllZerosRowCount(self, matrix: List[List[int]], mask: int) -> int:
    count = 0
    for row in matrix:
      isAllZeros = True
      for i, cell in enumerate(row):
        if cell == 1 and (mask >> i & 1) == 0:
          isAllZeros = False
          break
      if isAllZeros:
        count += 1
    return count


# Link: https://leetcode.com/problems/merge-nodes-in-between-zeros/description/
class Solution:
  def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
      return None
    if not head.next.val:
      node = ListNode(head.val)
      node.next = self.mergeNodes(head.next.next)
      return node

    next = self.mergeNodes(head.next)
    next.val += head.val
    return next


# Link: https://leetcode.com/problems/merge-nodes-in-between-zeros/description/
class Solution:
  def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
    curr = head.next

    while curr:
      running = curr
      summ = 0
      while running.val:
        summ += running.val
        running = running.next

      curr.val = summ
      curr.next = running.next
      curr = running.next

    return head.next


# Link: https://leetcode.com/problems/count-subarrays-with-more-ones-than-zeros/description/
class FenwichTree:
  def __init__(self, n: int):
    self.n = n
    self.sums = [0] * (2 * n + 1)

  def update(self, i: int, delta: int) -> None:
    i += self.n + 1  # re-mapping
    while i < len(self.sums):
      self.sums[i] += delta
      i += i & -i

  def get(self, i: int) -> int:
    i += self.n + 1  # re-mapping
    summ = 0
    while i > 0:
      summ += self.sums[i]
      i -= i & -i
    return summ


class Solution:
  def subarraysWithMoreZerosThanOnes(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    ans = 0
    prefix = 0
    tree = FenwichTree(len(nums))
    tree.update(0, 1)

    for num in nums:
      prefix += -1 if num == 0 else 1
      ans += tree.get(prefix - 1)
      ans %= kMod
      tree.update(prefix, 1)

    return ans


# Link: https://leetcode.com/problems/jump-game-viii/description/
class Solution:
  def minCost(self, nums: List[int], costs: List[int]) -> int:
    # dp[i] := the minimum cost to jump to i
    dp = [math.inf] * len(nums)
    maxStack = []
    minStack = []

    dp[0] = 0

    for i, num in enumerate(nums):
      while maxStack and num >= nums[maxStack[-1]]:
        dp[i] = min(dp[i], dp[maxStack.pop()] + costs[i])
      while minStack and num < nums[minStack[-1]]:
        dp[i] = min(dp[i], dp[minStack.pop()] + costs[i])
      maxStack.append(i)
      minStack.append(i)

    return dp[-1]


# Link: https://leetcode.com/problems/find-k-pairs-with-smallest-sums/description/
class Solution:
  def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    minHeap = []

    for i in range(min(k, len(nums1))):
      heapq.heappush(minHeap, (nums1[i] + nums2[0], i, 0))

    ans = []
    while minHeap and len(ans) < k:
      _, i, j = heapq.heappop(minHeap)
      ans.append([nums1[i], nums2[j]])
      if j + 1 < len(nums2):
        heapq.heappush(minHeap, (nums1[i] + nums2[j + 1], i, j + 1))

    return ans


# Link: https://leetcode.com/problems/ways-to-split-array-into-three-subarrays/description/
class Solution:
  def waysToSplit(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    ans = 0
    prefix = list(itertools.accumulate(nums))

    j = 0
    k = 0
    for i in range(n - 2):
      # Find the first index j s.t.
      # left = prefix[i] <= mid = prefix[j] - prefix[i]
      j = max(j, i + 1)
      while j < n - 1 and prefix[i] > prefix[j] - prefix[i]:
        j += 1
      # Find the first index k s.t.
      # mid = prefix[k] - prefix[i] > right = prefix[-1] - prefix[k]
      k = max(k, j)
      while k < n - 1 and prefix[k] - prefix[i] <= prefix[-1] - prefix[k]:
        k += 1
      ans += k - j
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/ways-to-split-array-into-three-subarrays/description/
class Solution:
  def waysToSplit(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    ans = 0
    prefix = list(itertools.accumulate(nums))

    def firstGreaterEqual(i: int) -> int:
      """Finds the first index j s.t.
         Mid = prefix[j] - prefix[i] >= left = prefix[i]
      """
      l = i + 1
      r = n - 1
      while l < r:
        m = (l + r) // 2
        if prefix[m] - prefix[i] >= prefix[i]:
          r = m
        else:
          l = m + 1
      return l

    def firstGreater(i: int) -> int:
      """Finds the first index k s.t.
         mid = prefix[k] - prefix[i] > right = prefix[-1] - prefix[k]
      """
      l = i + 1
      r = n - 1
      while l < r:
        m = (l + r) // 2
        if prefix[m] - prefix[i] > prefix[-1] - prefix[m]:
          r = m
        else:
          l = m + 1
      return l

    for i in range(n - 2):
      j = firstGreaterEqual(i)
      if j == n - 1:
        break
      mid = prefix[j] - prefix[i]
      right = prefix[-1] - prefix[j]
      if mid > right:
        continue
      k = firstGreater(i)
      ans = (ans + k - j) % kMod

    return ans


# Link: https://leetcode.com/problems/rank-teams-by-votes/description/
class Team:
  def __init__(self, name: str, teamSize: int):
    self.name = name
    self.rank = [0] * teamSize


class Solution:
  def rankTeams(self, votes: List[str]) -> str:
    teamSize = len(votes[0])
    teams = [Team(chr(ord('A') + i), teamSize) for i in range(26)]

    for vote in votes:
      for i in range(teamSize):
        teams[ord(vote[i]) - ord('A')].rank[i] += 1

    teams.sort(key=lambda x: (x.rank, -ord(x.name)), reverse=True)
    return ''.join(team.name for team in teams[:teamSize])


# Link: https://leetcode.com/problems/paint-fence/description/
class Solution:
  def numWays(self, n: int, k: int) -> int:
    if n == 0:
      return 0
    if n == 1:
      return k
    if n == 2:
      return k * k

    # dp[i] := the number of ways to pan posts with k colors
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = k
    dp[2] = k * k

    for i in range(3, n + 1):
      dp[i] = (dp[i - 1] + dp[i - 2]) * (k - 1)

    return dp[n]


# Link: https://leetcode.com/problems/delete-node-in-a-linked-list/description/
class Solution:
  def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next


# Link: https://leetcode.com/problems/unique-paths-ii/description/
class Solution:
  def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    dp = [0] * n
    dp[0] = 1

    for i in range(m):
      for j in range(n):
        if obstacleGrid[i][j]:
          dp[j] = 0
        elif j > 0:
          dp[j] += dp[j - 1]

    return dp[n - 1]


# Link: https://leetcode.com/problems/unique-paths-ii/description/
class Solution:
  def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    # dp[i][j] := the number of unique paths from (0, 0) to (i, j)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    dp[0][1] = 1  # Can also set dp[1][0] = 1.

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if obstacleGrid[i - 1][j - 1] == 0:
          dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m][n]


# Link: https://leetcode.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/description/
class Solution:
  def winnerOfGame(self, colors: str) -> bool:
    countAAA = 0
    countBBB = 0

    for a, b, c in zip(colors, colors[1:], colors[2:]):
      if 'A' == a == b == c:
        countAAA += 1
      elif 'B' == a == b == c:
        countBBB += 1

    return countAAA > countBBB


# Link: https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/description/
class Solution:
  def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0, head)
    slow = dummy
    fast = dummy

    while fast.next and fast.next.next:
      slow = slow.next
      fast = fast.next.next

    # Delete the middle node.
    slow.next = slow.next.next
    return dummy.next


# Link: https://leetcode.com/problems/number-of-spaces-cleaning-robot-cleaned/description/
class Solution:
  def numberOfCleanRooms(self, room: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(room)
    n = len(room[0])
    ans = 1
    i = 0
    j = 0
    state = 0  # 0 := right, 1 := down, 2 := left, 3 := up
    seen = {(i, j, state)}
    room[i][j] = 2  # 2 := cleaned

    while True:
      x = i + dirs[state]
      y = j + dirs[state + 1]
      if x < 0 or x == m or y < 0 or y == n or room[x][y] == 1:
        # Turn 90 degrees clockwise.
        state = (state + 1) % 4
      else:
        # Walk to (x, y).
        if room[x][y] == 0:
          ans += 1
          room[x][y] = 2
        i = x
        j = y
      if (x, y, state) in seen:
        return ans
      seen.add((x, y, state))


# Link: https://leetcode.com/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/description/
class Solution:
  def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
    n = len(edges) + 1
    tree = [[] for _ in range(n)]

    for u, v, w in edges:
      tree[u].append((v, w))
      tree[v].append((u, w))

    def connectablePairsRootedAt(u: int) -> int:
      pairs = 0
      count = 0
      for v, w in tree[u]:
        childCount = dfs(v, u, w)
        pairs += count * childCount
        count += childCount
      return pairs

    def dfs(u: int, prev: int, dist: int) -> int:
      return int(dist % signalSpeed == 0) + \
          sum(dfs(v, u, dist + w)
              for v, w in tree[u]
              if v != prev)

    return [connectablePairsRootedAt(i) for i in range(n)]


# Link: https://leetcode.com/problems/maximum-of-absolute-value-expression/description/
class Solution:
  def maxAbsValExpr(self, arr1: List[int], arr2: List[int]) -> int:
    n = len(arr1)
    a = [arr1[i] + arr2[i] + i for i in range(n)]
    b = [arr1[i] + arr2[i] - i for i in range(n)]
    c = [arr1[i] - arr2[i] + i for i in range(n)]
    d = [arr1[i] - arr2[i] - i for i in range(n)]
    return max(map(lambda x: max(x) - min(x), (a, b, c, d)))


# Link: https://leetcode.com/problems/using-a-robot-to-print-the-lexicographically-smallest-string/description/
class Solution:
  def robotWithString(self, s: str) -> str:
    ans = []
    count = collections.Counter(s)
    stack = []

    for c in s:
      stack.append(c)
      count[c] -= 1
      minChar = self._getMinChar(count)
      while stack and stack[-1] <= minChar:
        ans.append(stack.pop())

    return ''.join(ans + stack[::-1])

  def _getMinChar(self, count: List[int]) -> str:
    for c in string.ascii_lowercase:
      if count[c]:
        return c
    return 'a'


# Link: https://leetcode.com/problems/the-number-of-full-rounds-you-have-played/description/
class Solution:
  def numberOfRounds(self, loginTime: str, logoutTime: str) -> int:
    start = self._getMinutes(loginTime)
    finish = self._getMinutes(logoutTime)
    if start > finish:
      finish += 60 * 24

    return max(0, finish // 15 - (start + 14) // 15)

  def _getMinutes(self, time: str) -> int:
    h, m = map(int, time.split(':'))
    return 60 * h + m


# Link: https://leetcode.com/problems/shortest-and-lexicographically-smallest-beautiful-string/description/
class Solution:
  # Same as 76. Minimum Window Substring
  def shortestBeautifulSubstring(self, s: str, k: int) -> str:
    bestLeft = -1
    minLength = len(s) + 1
    ones = 0

    l = 0
    for r, c in enumerate(s):
      if c == '1':
        ones += 1
      while ones == k:
        if r - l + 1 < minLength:
          bestLeft = l
          minLength = r - l + 1
        elif r - l + 1 == minLength and s[l:l + minLength] < s[bestLeft:bestLeft + minLength]:
          bestLeft = l
        if s[l] == '1':
          ones -= 1
        l += 1

    return "" if bestLeft == -1 else s[bestLeft:bestLeft + minLength]


# Link: https://leetcode.com/problems/house-robber-iii/description/
class Solution:
  def rob(self, root: Optional[TreeNode]) -> int:
    def robOrNot(root: Optional[TreeNode]) -> tuple:
      if not root:
        return (0, 0)

      robLeft, notRobLeft = robOrNot(root.left)
      robRight, notRobRight = robOrNot(root.right)

      return (root.val + notRobLeft + notRobRight,
              max(robLeft, notRobLeft) + max(robRight, notRobRight))

    return max(robOrNot(root))


# Link: https://leetcode.com/problems/number-of-ways-to-build-house-of-cards/description/
class Solution:
  def houseOfCards(self, n: int) -> int:
    # dp[i] := the number of valid result for i cards
    dp = [1] + [0] * n

    for baseCards in range(2, n + 1, 3):
      for i in range(n, baseCards - 1, -1):
        # Use `baseCards` as the base, so we're left with `i - baseCards` cards.
        dp[i] += dp[i - baseCards]

    return dp[n]


# Link: https://leetcode.com/problems/push-dominoes/description/
class Solution:
  def pushDominoes(self, dominoes: str) -> str:
    ans = list(dominoes)
    L = -1
    R = -1

    for i in range(len(dominoes) + 1):
      if i == len(dominoes) or dominoes[i] == 'R':
        if L < R:
          while R < i:
            ans[R] = 'R'
            R += 1
        R = i
      elif dominoes[i] == 'L':
        if R < L or (L, R) == (-1, -1):
          if (L, R) == (-1, -1):
            L += 1
          while L < i:
            ans[L] = 'L'
            L += 1
        else:
          l = R + 1
          r = i - 1
          while l < r:
            ans[l] = 'R'
            ans[r] = 'L'
            l += 1
            r -= 1
        L = i

    return ''.join(ans)


# Link: https://leetcode.com/problems/removing-minimum-number-of-magic-beans/description/
class Solution:
  def minimumRemoval(self, beans: List[int]) -> int:
    n = len(beans)
    summ = sum(beans)
    return min(summ - (n - i) * bean
               for i, bean in enumerate(sorted(beans)))


# Link: https://leetcode.com/problems/design-snake-game/description/
class SnakeGame:
  def __init__(self, width: int, height: int, food: List[List[int]]):
    """
    Initialize your data structure here.
    @param width - screen width
    @param height - screen height
    @param food - A list of food positions
    E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
    """
    self.width = width
    self.height = height
    self.food = food
    self.score = 0
    self.k = 0  # food's index
    self.lookup = set([self.getId(0, 0)])
    self.body = collections.deque([self.getId(0, 0)])  # snake's body

  def move(self, direction: str) -> int:
    """
    Moves the snake.
    @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
    @return The game's score after the move. Return -1 if game over.
    Game over when snake crosses the screen boundary or bites its body.
    """
    # the old head's position
    i = self.body[0] // self.width
    j = self.body[0] % self.width

    # Update the head's position and check if it's out-of-bounds.
    if direction == "U":
      i -= 1
      if i < 0:
        return -1
    if direction == "L":
      j -= 1
      if j < 0:
        return -1
    if direction == "R":
      j += 1
      if j == self.width:
        return -1
    if direction == "D":
      i += 1
      if i == self.height:
        return -1

    newHead = self.getId(i, j)

    # Case 1: Eat food and increase the size by 1.
    if self.k < len(self.food) and i == self.food[self.k][0] and j == self.food[self.k][1]:
      self.lookup.add(newHead)
      self.body.appendleft(newHead)
      self.k += 1
      self.score += 1
      return self.score

    # Case 2: new head != old tail and eat body!
    if newHead != self.body[-1] and newHead in self.lookup:
      return -1

    # Case 3: normal case
    # Remove the old tail first, then add new head because new head may be in
    # old tail's position.
    self.lookup.remove(self.body[-1])
    self.lookup.add(newHead)
    self.body.pop()
    self.body.appendleft(newHead)

    return self.score

  def getId(self, i: int, j: int) -> int:
    return i * self.width + j


# Link: https://leetcode.com/problems/total-hamming-distance/description/
class Solution:
  def totalHammingDistance(self, nums: List[int]) -> int:
    kMaxBit = 30
    ans = 0

    for i in range(kMaxBit):
      ones = sum(num & (1 << i) > 0 for num in nums)
      zeros = len(nums) - ones
      ans += ones * zeros

    return ans


# Link: https://leetcode.com/problems/circular-permutation-in-binary-representation/description/
class Solution:
  def circularPermutation(self, n: int, start: int) -> List[int]:
    return [start ^ i ^ i >> 1 for i in range(1 << n)]


# Link: https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/description/
class Solution:
  def lenLongestFibSubseq(self, arr: List[int]) -> int:
    n = len(arr)
    ans = 0
    numToIndex = {a: i for i, a in enumerate(arr)}
    dp = [[2] * n for _ in range(n)]

    for j in range(n):
      for k in range(j + 1, n):
        ai = arr[k] - arr[j]
        if ai < arr[j] and ai in numToIndex:
          i = numToIndex[ai]
          dp[j][k] = dp[i][j] + 1
          ans = max(ans, dp[j][k])

    return ans


# Link: https://leetcode.com/problems/count-collisions-on-a-road/description/
class Solution:
  def countCollisions(self, directions: str) -> int:
    l = 0
    r = len(directions) - 1

    while l < len(directions) and directions[l] == 'L':
      l += 1

    while r >= 0 and directions[r] == 'R':
      r -= 1

    return sum(c != 'S' for c in directions[l:r + 1])


# Link: https://leetcode.com/problems/removing-minimum-and-maximum-from-array/description/
class Solution:
  def minimumDeletions(self, nums: List[int]) -> int:
    n = len(nums)
    a = nums.index(min(nums))
    b = nums.index(max(nums))
    if a > b:
      a, b = b, a
    return min(a + 1 + n - b, b + 1, n - a)


# Link: https://leetcode.com/problems/string-compression/description/
class Solution:
  def compress(self, chars: List[str]) -> int:
    ans = 0
    i = 0

    while i < len(chars):
      letter = chars[i]
      count = 0
      while i < len(chars) and chars[i] == letter:
        count += 1
        i += 1
      chars[ans] = letter
      ans += 1
      if count > 1:
        for c in str(count):
          chars[ans] = c
          ans += 1

    return ans


# Link: https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/description/
class Solution:
  def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
    ans = -1
    graph = self._getGraph(root)
    q = collections.deque([start])
    seen = {start}

    while q:
      ans += 1
      for _ in range(len(q)):
        u = q.popleft()
        if u not in graph:
          continue
        for v in graph[u]:
          if v in seen:
            continue
          q.append(v)
          seen.add(v)

    return ans

  def _getGraph(self, root: Optional[TreeNode]) -> Dict[int, List[int]]:
    graph = collections.defaultdict(list)
    q = collections.deque([(root, -1)])  # (node, parent)

    while q:
      node, parent = q.popleft()
      if parent != -1:
        graph[parent].append(node.val)
        graph[node.val].append(parent)
      if node.left:
        q.append((node.left, node.val))
      if node.right:
        q.append((node.right, node.val))

    return graph


# Link: https://leetcode.com/problems/maximum-score-from-removing-substrings/description/
class Solution:
  def maximumGain(self, s: str, x: int, y: int) -> int:
    # The assumption that gain('ab') > gain('ba') while removing 'ba' first is
    # optimal is contradicted. Only 'b(ab)a' satisfies the condition of
    # preventing two 'ba' removals, but after removing 'ab', we can still
    # remove one 'ba', resulting in a higher gain. Thus, removing 'ba' first is
    # not optimal.
    return self._gain(s, 'ab', x, 'ba', y) if x > y \
        else self._gain(s, 'ba', y, 'ab', x)

  # Returns the points gained by first removing sub1 ('ab' | 'ba') from s with
  # point1, then removing sub2 ('ab' | 'ba') from s with point2.
  def _gain(self, s: str, sub1: str, point1: int, sub2: str, point2: int) -> int:
    points = 0
    stack1 = []
    stack2 = []

    # Remove 'sub1' from s with point1 gain.
    for c in s:
      if stack1 and stack1[-1] == sub1[0] and c == sub1[1]:
        stack1.pop()
        points += point1
      else:
        stack1.append(c)

    # Remove 'sub2' from s with point2 gain.
    for c in stack1:
      if stack2 and stack2[-1] == sub2[0] and c == sub2[1]:
        stack2.pop()
        points += point2
      else:
        stack2.append(c)

    return points


# Link: https://leetcode.com/problems/take-k-of-each-character-from-left-and-right/description/
class Solution:
  def takeCharacters(self, s: str, k: int) -> int:
    n = len(s)
    ans = n
    count = collections.Counter(s)
    if any(count[c] < k for c in 'abc'):
      return -1

    l = 0
    for r, c in enumerate(s):
      count[c] -= 1
      while count[c] < k:
        count[s[l]] += 1
        l += 1
      ans = min(ans, n - (r - l + 1))

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-frogs-croaking/description/
class Solution:
  def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
    kCroak = 'croak'
    ans = 0
    frogs = 0
    count = [0] * 5

    for c in croakOfFrogs:
      count[kCroak.index(c)] += 1
      if any(count[i] > count[i - 1] for i in range(1, 5)):
        return -1
      if c == 'c':
        frogs += 1
      elif c == 'k':
        frogs -= 1
      ans = max(ans, frogs)

    return ans if frogs == 0 else -1


# Link: https://leetcode.com/problems/interval-list-intersections/description/
class Solution:
  def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    ans = []
    i = 0
    j = 0

    while i < len(firstList) and j < len(secondList):
      # lo := the start of the intersection
      # hi := the end of the intersection
      lo = max(firstList[i][0], secondList[j][0])
      hi = min(firstList[i][1], secondList[j][1])
      if lo <= hi:
        ans.append([lo, hi])
      if firstList[i][1] < secondList[j][1]:
        i += 1
      else:
        j += 1

    return ans


# Link: https://leetcode.com/problems/design-a-number-container-system/description/
from sortedcontainers import SortedSet


class NumberContainers:
  def __init__(self):
    self.numberToIndices = collections.defaultdict(SortedSet)
    self.indexToNumber = {}

  def change(self, index: int, number: int) -> None:
    if index in self.indexToNumber:
      originalNumber = self.indexToNumber[index]
      self.numberToIndices[originalNumber].remove(index)
      if len(self.numberToIndices[originalNumber]) == 0:
        del self.numberToIndices[originalNumber]
    self.indexToNumber[index] = number
    self.numberToIndices[number].add(index)

  def find(self, number: int) -> int:
    if number in self.numberToIndices:
      return self.numberToIndices[number][0]
    return -1


# Link: https://leetcode.com/problems/map-of-highest-peak/description/
class Solution:
  def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(isWater)
    n = len(isWater[0])
    ans = [[-1] * n for _ in range(m)]
    q = collections.deque()

    for i in range(m):
      for j in range(n):
        if isWater[i][j] == 1:
          q.append((i, j))
          ans[i][j] = 0

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if ans[x][y] != -1:
          continue
        ans[x][y] = ans[i][j] + 1
        q.append((x, y))

    return ans


# Link: https://leetcode.com/problems/make-k-subarray-sums-equal/description/
class Solution:
  def makeSubKSumEqual(self, arr: List[int], k: int) -> int:
    # If the sum of each subarray of length k is equal, then `arr` must have a
    # repeated pattern of size k. e.g. arr = [1, 2, 3, ...] and k = 3, to have
    # sum([1, 2, 3)] == sum([2, 3, x]), x must be 1. Therefore, arr[i] ==
    # arr[(i + k) % n] for every i.
    n = len(arr)
    ans = 0
    seen = [0] * n

    for i in range(n):
      groups = []
      j = i
      while not seen[j]:
        groups.append(arr[j])
        seen[j] = True
        j = (j + k) % n
      groups.sort()
      for num in groups:
        ans += abs(num - groups[len(groups) // 2])

    return ans


# Link: https://leetcode.com/problems/is-array-a-preorder-of-some-binary-tree/description/
class Solution:
  def isPreorder(self, nodes: List[List[int]]) -> bool:
    stack = []  # Stores `id`s.

    for id, parentId in nodes:
      if parentId == -1:
        stack.append(id)
        continue
      while stack and stack[-1] != parentId:
        stack.pop()
      if not stack:
        return False
      stack.append(id)

    return True


# Link: https://leetcode.com/problems/maximum-xor-product/description/
class Solution:
  def maximumXorProduct(self, a: int, b: int, n: int) -> int:
    kMod = 1_000_000_007
    for bit in (2**i for i in range(n)):
      # Pick a bit if it makes min(a, b) larger.
      if a * b < (a ^ bit) * (b ^ bit):
        a ^= bit
        b ^= bit
    return a * b % kMod


# Link: https://leetcode.com/problems/frequency-of-the-most-frequent-element/description/
class Solution:
  def maxFrequency(self, nums: List[int], k: int) -> int:
    ans = 0
    summ = 0

    nums.sort()

    l = 0
    for r, num in enumerate(nums):
      summ += num
      while summ + k < num * (r - l + 1):
        summ -= nums[l]
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/description/
class Solution:
  def minSteps(self, s: str, t: str) -> int:
    count = collections.Counter(s)
    count.subtract(collections.Counter(t))
    return sum(abs(value) for value in count.values()) // 2


# Link: https://leetcode.com/problems/edit-distance/description/
class Solution:
  def minDistance(self, word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    # dp[i][j] := the minimum number of operations to convert word1[0..i) to
    # word2[0..j)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
      dp[i][0] = i

    for j in range(1, n + 1):
      dp[0][j] = j

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if word1[i - 1] == word2[j - 1]:
          dp[i][j] = dp[i - 1][j - 1]
        else:
          dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    return dp[m][n]


# Link: https://leetcode.com/problems/compare-version-numbers/description/
class Solution:
  def compareVersion(self, version1: str, version2: str) -> int:
    levels1 = version1.split('.')
    levels2 = version2.split('.')
    length = max(len(levels1), len(levels2))

    for i in range(length):
      v1 = int(levels1[i]) if i < len(levels1) else 0
      v2 = int(levels2[i]) if i < len(levels2) else 0
      if v1 < v2:
        return -1
      if v1 > v2:
        return 1

    return 0


# Link: https://leetcode.com/problems/walking-robot-simulation-ii/description/
class Robot:
  def __init__(self, width: int, height: int):
    self.isOrigin = True
    self.i = 0
    self.pos = [((0, 0), 'South')] + \
        [((i, 0), 'East') for i in range(1, width)] + \
        [((width - 1, j), 'North') for j in range(1, height)] + \
        [((i, height - 1), 'West') for i in range(width - 2, -1, -1)] +\
        [((0, j), 'South') for j in range(height - 2, 0, -1)]

  def step(self, num: int) -> None:
    self.isOrigin = False
    self.i = (self.i + num) % len(self.pos)

  def getPos(self) -> List[int]:
    return self.pos[self.i][0]

  def getDir(self) -> str:
    return 'East' if self.isOrigin else self.pos[self.i][1]


# Link: https://leetcode.com/problems/last-stone-weight-ii/description/
class Solution:
  def lastStoneWeightII(self, stones: List[int]) -> int:
    summ = sum(stones)
    s = 0
    dp = [True] + [False] * summ

    for stone in stones:
      for w in range(summ // 2 + 1)[::-1]:
        if w >= stone:
          dp[w] = dp[w] or dp[w - stone]
        if dp[w]:
          s = max(s, w)

    return summ - 2 * s


# Link: https://leetcode.com/problems/guess-the-number-using-bitwise-questions-i/description/
# Definition of commonSetBits API.
# def commonSetBits(num: int) -> int:

class Solution:
  def findNumber(self) -> int:
    return sum(1 << i for i in range(31)
               if commonSetBits(1 << i) == 1)


# Link: https://leetcode.com/problems/maximum-average-pass-ratio/description/
class Solution:
  def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
    def extraPassRatio(pas: int, total: int) -> float:
      """Returns the extra pass ratio if a brilliant student joins."""
      return (pas + 1) / (total + 1) - pas / total

    maxHeap = [(-extraPassRatio(pas, total), pas, total) for pas, total in classes]
    heapq.heapify(maxHeap)

    for _ in range(extraStudents):
      _, pas, total = heapq.heappop(maxHeap)
      heapq.heappush(
          maxHeap, (-extraPassRatio(pas + 1, total + 1), pas + 1, total + 1))

    return sum(pas / total for _, pas, total in maxHeap) / len(maxHeap)


# Link: https://leetcode.com/problems/minimum-cost-to-make-array-equalindromic/description/
class Solution:
  def minimumCost(self, nums: List[int]) -> int:
    nums.sort()
    median = nums[len(nums) // 2]
    nextPalindrome = self._getPalindrome(median, delta=1)
    prevPalindrome = self._getPalindrome(median, delta=-1)
    return min(self._cost(nums, nextPalindrome),
               self._cost(nums, prevPalindrome))

  def _cost(self, nums: List[int], palindrome: int) -> int:
    """Returns the cost to change all the numbers to `palindrome`."""
    return sum(abs(palindrome - num) for num in nums)

  def _getPalindrome(self, num: int, delta: int) -> int:
    """Returns the palindrome `p`, where p = num + a * delta and a > 0."""
    while not self._isPalindrome(num):
      num += delta
    return num

  def _isPalindrome(self, num: int) -> int:
    original = str(num)
    return original == original[::-1]


# Link: https://leetcode.com/problems/maximum-beauty-of-an-array-after-applying-operation/description/
class Solution:
  def maximumBeauty(self, nums: List[int], k: int) -> int:
    # l and r track the maximum window instead of the valid window.
    nums.sort()

    l = 0
    for r in range(len(nums)):
      if nums[r] - nums[l] > 2 * k:
        l += 1

    return r - l + 1


# Link: https://leetcode.com/problems/maximum-beauty-of-an-array-after-applying-operation/description/
class Solution:
  def maximumBeauty(self, nums: List[int], k: int) -> int:
    ans = 0

    nums.sort()

    l = 0
    for r in range(len(nums)):
      while nums[r] - nums[l] > 2 * k:
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/copy-list-with-random-pointer/description/
class Solution:
  def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
      return None
    if head in self.map:
      return self.map[head]

    newNode = Node(head.val)
    self.map[head] = newNode
    newNode.next = self.copyRandomList(head.next)
    newNode.random = self.copyRandomList(head.random)
    return newNode

  map = {}


# Link: https://leetcode.com/problems/jump-game-vi/description/
class Solution:
  def maxResult(self, nums: List[int], k: int) -> int:
    # Stores dp[i] within the bounds.
    maxQ = collections.deque([0])
    # dp[i] := the maximum score to consider nums[0..i]
    dp = [0] * len(nums)
    dp[0] = nums[0]

    for i in range(1, len(nums)):
      # Pop the index if it's out-of-bounds.
      if maxQ[0] + k < i:
        maxQ.popleft()
      dp[i] = dp[maxQ[0]] + nums[i]
      # Pop indices that won't be chosen in the future.
      while maxQ and dp[maxQ[-1]] <= dp[i]:
        maxQ.pop()
      maxQ.append(i)

    return dp[-1]


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits/description/
class Solution:
  def minimumCoins(self, prices: List[int]) -> int:
    n = len(prices)
    # Convert to 0-indexed for easy computation.
    # dp[i] := the minimum number of coins to acquire fruits[i:]
    dp = [math.inf] * n + [0]

    for i in range(n - 1, -1, -1):
      # Convert back to 1-indexed.
      for j in range(i + 1, min((i + 1) * 2 + 1, n + 1)):
        dp[i] = min(dp[i], prices[i] + dp[j])

    return dp[0]


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits/description/
class Solution:
  def minimumCoins(self, prices: List[int]) -> int:
    n = len(prices)
    # Stores (dp[i], i), where dp[i] is the minimum number of coins to acquire
    # fruits[i:] (0-indexed).
    minHeap = [(0, n)]
    ans = 0

    for i in range(n - 1, -1, -1):
      while minHeap and minHeap[0][1] > (i + 1) * 2:
        heapq.heappop(minHeap)
      ans = prices[i] + minHeap[0][0]
      heapq.heappush(minHeap, (ans, i))

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits/description/
class Solution:
  def minimumCoins(self, prices: List[int]) -> int:
    n = len(prices)
    ans = math.inf
    # Stores (dp[i], i), where dp[i] := the minimum number of coins to acquire
    # fruits[i:] (0-indexed) in ascending order.
    minQ = collections.deque([(0, n)])

    for i in range(n - 1, -1, -1):
      while minQ and minQ[0][1] > (i + 1) * 2:
        minQ.popleft()
      ans = prices[i] + minQ[0][0]
      while minQ and minQ[-1][0] >= ans:
        minQ.pop()
      minQ.append((ans, i))

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-separate-sentence-into-rows/description/
class Solution:
  def minimumCost(self, sentence: str, k: int) -> int:
    if len(sentence) <= k:
      return 0

    words = sentence.split()

    # dp[i] := the minimum cost of the first i words
    dp = [0] * (len(words) + 1)

    for i in range(1, len(words) + 1):
      n = len(words[i - 1])  # the length of the current row
      dp[i] = dp[i - 1] + (k - n)**2
      # Gradually add words[j - 1], words[j - 2], ....
      for j in range(i - 1, 0, -1):
        n += len(words[j - 1]) + 1
        if n > k:
          break
        dp[i] = min(dp[i], dp[j - 1] + (k - n)**2)

    lastRowLen = len(words[-1])
    i = len(words) - 2  # Greedily put words into last row

    while i > 0 and lastRowLen + len(words[i]) + 1 <= k:
      lastRowLen += len(words[i]) + 1
      i -= 1

    return min(dp[i + 1:len(words)])


# Link: https://leetcode.com/problems/queries-on-number-of-points-inside-a-circle/description/
class Solution:
  def countPoints(self, points: List[List[int]], queries: List[List[int]]) -> List[int]:
    ans = []

    for xj, yj, rj in queries:
      count = 0
      for xi, yi in points:
        if (xi - xj)**2 + (yi - yj)**2 <= rj**2:
          count += 1
      ans.append(count)

    return ans


# Link: https://leetcode.com/problems/range-sum-query-2d-immutable/description/
class NumMatrix:
  def __init__(self, matrix: List[List[int]]):
    if not matrix:
      return

    m = len(matrix)
    n = len(matrix[0])
    # prefix[i][j] := the sum of matrix[0..i)[0..j)
    self.prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        self.prefix[i + 1][j + 1] = \
            matrix[i][j] + self.prefix[i][j + 1] + \
            self.prefix[i + 1][j] - self.prefix[i][j]

  def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
    return self.prefix[row2 + 1][col2 + 1] - self.prefix[row1][col2 + 1] - \
        self.prefix[row2 + 1][col1] + self.prefix[row1][col1]


# Link: https://leetcode.com/problems/repeated-dna-sequences/description/
class Solution:
  def findRepeatedDnaSequences(self, s: str) -> List[str]:
    ans = set()
    seen = set()

    for i in range(len(s) - 9):
      seq = s[i:i + 10]
      if seq in seen:
        ans.add(seq)
      seen.add(seq)

    return list(ans)


# Link: https://leetcode.com/problems/grumpy-bookstore-owner/description/
class Solution:
  def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
    satisfied = sum(c for i, c in enumerate(customers) if grumpy[i] == 0)
    madeSatisfied = 0
    windowSatisfied = 0

    for i, customer in enumerate(customers):
      if grumpy[i] == 1:
        windowSatisfied += customer
      if i >= X and grumpy[i - X] == 1:
        windowSatisfied -= customers[i - X]
      madeSatisfied = max(madeSatisfied, windowSatisfied)

    return satisfied + madeSatisfied


# Link: https://leetcode.com/problems/evaluate-division/description/
class Solution:
  def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    ans = []
    # graph[A][B] := A / B
    graph = collections.defaultdict(dict)

    for (A, B), value in zip(equations, values):
      graph[A][B] = value
      graph[B][A] = 1 / value

    def devide(A: str, C: str, seen: Set[str]) -> float:
      """Returns A / C."""
      if A == C:
        return 1.0

      seen.add(A)

      # value := A / B
      for B, value in graph[A].items():
        if B in seen:
          continue
        res = devide(B, C, seen)  # B / C
        if res > 0:  # valid result
          return value * res  # (A / B) * (B / C) = A / C

      return -1.0  # invalid result

    for A, C in queries:
      if A not in graph or C not in graph:
        ans.append(-1.0)
      else:
        ans.append(devide(A, C, set()))

    return ans


# Link: https://leetcode.com/problems/campus-bikes-ii/description/
class Solution:
  def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
    def dist(p1: List[int], p2: List[int]) -> int:
      return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @functools.lru_cache(None)
    def dp(workerIndex: int, used: int) -> int:
      """
      Returns the minimum Manhattan distances to assign bikes to
      workers[workerIndex..n), where `used` is the bitmask of the used bikes.
      """
      if workerIndex == len(workers):
        return 0
      return min((dist(workers[workerIndex], bike) + dp(workerIndex + 1, used | 1 << i)
                  for i, bike in enumerate(bikes)
                  if not used & 1 << i), default=math.inf)

    return dp(0, 0)


# Link: https://leetcode.com/problems/design-memory-allocator/description/
class Allocator:
  def __init__(self, n: int):
    self.memory = [0] * n
    self.mIDToIndices = [[] for _ in range(1001)]

  def allocate(self, size: int, mID: int) -> int:
    consecutiveFree = 0
    for i, m in enumerate(self.memory):
      consecutiveFree = consecutiveFree + 1 if m == 0 else 0
      if consecutiveFree == size:
        for j in range(i - consecutiveFree + 1, i + 1):
          self.memory[j] = mID
          self.mIDToIndices[mID].append(j)
        return i - consecutiveFree + 1
    return -1

  def free(self, mID: int) -> int:
    indices = self.mIDToIndices[mID]
    freedUnits = len(indices)
    for index in indices:
      self.memory[index] = 0
    indices.clear()
    return freedUnits


# Link: https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/description/
class Solution:
  def countPaths(self, n: int, roads: List[List[int]]) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in roads:
      graph[u].append((v, w))
      graph[v].append((u, w))

    return self._dijkstra(graph, 0, n - 1)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, dst: int) -> int:
    kMod = 10**9 + 7
    ways = [0] * len(graph)
    dist = [math.inf] * len(graph)

    ways[src] = 1
    dist[src] = 0
    minHeap = [(dist[src], src)]

    while minHeap:
      d, u = heapq.heappop(minHeap)
      if d > dist[u]:
        continue
      for v, w in graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          ways[v] = ways[u]
          heapq.heappush(minHeap, (dist[v], v))
        elif d + w == dist[v]:
          ways[v] += ways[u]
          ways[v] %= kMod

    return ways[dst]


# Link: https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/description/
class Solution:
  def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
    inDegrees = [0] * n

    for _, v in edges:
      inDegrees[v] += 1

    return [i for i, d in enumerate(inDegrees) if d == 0]


# Link: https://leetcode.com/problems/minimum-cost-to-connect-sticks/description/
class Solution:
  def connectSticks(self, sticks: List[int]) -> int:
    ans = 0
    heapq.heapify(sticks)

    while len(sticks) > 1:
      x = heapq.heappop(sticks)
      y = heapq.heappop(sticks)
      ans += x + y
      heapq.heappush(sticks, x + y)

    return ans


# Link: https://leetcode.com/problems/game-of-life/description/
class Solution:
  def gameOfLife(self, board: List[List[int]]) -> None:
    m = len(board)
    n = len(board[0])

    for i in range(m):
      for j in range(n):
        ones = 0
        for x in range(max(0, i - 1), min(m, i + 2)):
          for y in range(max(0, j - 1), min(n, j + 2)):
            ones += board[x][y] & 1
        # Any live cell with two or three live neighbors lives on to the next
        # generation.
        if board[i][j] == 1 and (ones == 3 or ones == 4):
          board[i][j] |= 0b10
        # Any dead cell with exactly three live neighbors becomes a live cell,
        # as if by reproduction.
        if board[i][j] == 0 and ones == 3:
          board[i][j] |= 0b10

    for i in range(m):
      for j in range(n):
        board[i][j] >>= 1


# Link: https://leetcode.com/problems/next-greater-element-ii/description/
class Solution:
  def nextGreaterElements(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = [-1] * n
    stack = []  # a decreasing stack storing indices

    for i in range(n * 2):
      num = nums[i % n]
      while stack and nums[stack[-1]] < num:
        ans[stack.pop()] = num
      if i < n:
        stack.append(i)

    return ans


# Link: https://leetcode.com/problems/even-odd-tree/description/
class Solution:
  def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
    q = collections.deque([root])
    isEven = True

    while q:
      prevVal = -math.inf if isEven else math.inf
      for _ in range(sz):
        node = q.popleft()
        if isEven and (node.val % 2 == 0 or node.val <= prevVal):
          return False  # invalid case on even level
        if not isEven and (node.val % 2 == 1 or node.val >= prevVal):
          return False  # invalid case on odd level
        prevVal = node.val
        if node.left:
          q.append(node.left)
        if node.right:
          q.append(node.right)
      isEven = not isEven

    return True


# Link: https://leetcode.com/problems/clone-graph/description/
class Solution:
  def cloneGraph(self, node: 'Node') -> 'Node':
    if not node:
      return None

    q = collections.deque([node])
    map = {node: Node(node.val)}

    while q:
      u = q.popleft()
      for v in u.neighbors:
        if v not in map:
          map[v] = Node(v.val)
          q.append(v)
        map[u].neighbors.append(map[v])

    return map[node]


# Link: https://leetcode.com/problems/clone-graph/description/
class Solution:
  def cloneGraph(self, node: 'Node') -> 'Node':
    if not node:
      return None
    if node in self.map:
      return self.map[node]

    newNode = Node(node.val, [])
    self.map[node] = newNode

    for neighbor in node.neighbors:
      self.map[node].neighbors.append(self.cloneGraph(neighbor))

    return newNode

  map = {}


# Link: https://leetcode.com/problems/decode-the-slanted-ciphertext/description/
class Solution:
  def decodeCiphertext(self, encodedText: str, rows: int) -> str:
    n = len(encodedText)
    cols = n // rows

    ans = []
    matrix = [[' '] * cols for _ in range(rows)]

    for i in range(rows):
      for j in range(cols):
        matrix[i][j] = encodedText[i * cols + j]

    for col in range(cols):
      i = 0
      j = col
      while i < rows and j < cols:
        ans.append(matrix[i][j])
        i += 1
        j += 1

    return ''.join(ans).rstrip()


# Link: https://leetcode.com/problems/decode-the-slanted-ciphertext/description/
class Solution:
  def decodeCiphertext(self, encodedText: str, rows: int) -> str:
    n = len(encodedText)
    cols = n // rows

    ans = []

    for j in range(cols):
      for i in range(j, n, cols + 1):
        ans.append(encodedText[i])

    return ''.join(ans).rstrip()


# Link: https://leetcode.com/problems/flip-equivalent-binary-trees/description/
class Solution:
  def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    if not root1:
      return not root2
    if not root2:
      return not root1
    if root1.val != root2.val:
      return False
    return self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right) or \
        self.flipEquiv(root1.left, root2.right) and self.flipEquiv(
        root1.right, root2.left)


# Link: https://leetcode.com/problems/max-consecutive-ones-ii/description/
class Solution:
  def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    ans = 0
    zeros = 0

    l = 0
    for r, num in enumerate(nums):
      if num == 0:
        zeros += 1
      while zeros == 2:
        if nums[l] == 0:
          zeros -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/max-consecutive-ones-ii/description/
class Solution:
  def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    ans = 0
    lastZeroIndex = -1

    l = 0
    for r, num in enumerate(nums):
      if num == 0:
        l = lastZeroIndex + 1
        lastZeroIndex = r
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/max-consecutive-ones-ii/description/
class Solution:
  def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    maxZeros = 1
    ans = 0
    q = collections.deque()  # Store indices of zero.

    l = 0
    for r, num in enumerate(nums):
      if num == 0:
        q.append(r)
      if len(q) > maxZeros:
        l = q.popleft() + 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/find-the-derangement-of-an-array/description/
class Solution:
  def findDerangement(self, n: int) -> int:
    kMod = 1_000_000_007
    dp = [1] + [0] * n

    for i in range(2, n + 1):
      dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2]) % kMod

    return dp[n]


# Link: https://leetcode.com/problems/find-the-derangement-of-an-array/description/
class Solution:
  def findDerangement(self, n: int) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(i: int) -> int:
      if i == 0:
        return 1
      if i == 1:
        return 0
      return (i - 1) * (dp(i - 1) + dp(i - 2)) % kMod

    return dp(n)


# Link: https://leetcode.com/problems/largest-submatrix-with-rearrangements/description/
class Solution:
  def largestSubmatrix(self, matrix: List[List[int]]) -> int:
    ans = 0
    hist = [0] * len(matrix[0])

    for row in matrix:
      # Accumulate the histogram if possible.
      for i, num in enumerate(row):
        hist[i] = 0 if num == 0 else hist[i] + 1

      # Get the sorted histogram.
      sortedHist = sorted(hist)

      # Greedily calculate the answer.
      for i, h in enumerate(sortedHist):
        ans = max(ans, h * (len(row) - i))

    return ans


# Link: https://leetcode.com/problems/shortest-unsorted-continuous-subarray/description/
class Solution:
  def findUnsortedSubarray(self, nums: List[int]) -> int:
    mini = math.inf
    maxi = -math.inf
    flag = False

    for i in range(1, len(nums)):
      if nums[i] < nums[i - 1]:
        flag = True
      if flag:
        mini = min(mini, nums[i])

    flag = False

    for i in reversed(range(len(nums) - 1)):
      if nums[i] > nums[i + 1]:
        flag = True
      if flag:
        maxi = max(maxi, nums[i])

    for l in range(len(nums)):
      if nums[l] > mini:
        break

    for r, num in reversed(list(enumerate(nums))):
      if num < maxi:
        break

    return 0 if l >= r else r - l + 1


# Link: https://leetcode.com/problems/maximum-ice-cream-bars/description/
class Solution:
  def maxIceCream(self, costs: List[int], coins: int) -> int:
    for i, cost in enumerate(sorted(costs)):
      if coins >= cost:
        coins -= cost
      else:
        return i

    return len(costs)


# Link: https://leetcode.com/problems/check-knight-tour-configuration/description/
class Solution:
  def checkValidGrid(self, grid: List[List[int]]) -> bool:
    dirs = ((1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    n = len(grid)
    i = 0
    j = 0

    def nextGrid(i: int, j: int, target: int) -> Tuple[int, int]:
      """
      Returns (x, y), where grid[x][y] == target if (i, j) can reach target.
      """
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x >= n or y < 0 or y >= n:
          continue
        if grid[x][y] == target:
          return (x, y)
      return (-1, -1)

    for target in range(1, n * n):
      x, y = nextGrid(i, j, target)
      if x == -1 and y == -1:
        return False
      # Move (x, y) to (i, j).
      i = x
      j = y

    return True


# Link: https://leetcode.com/problems/number-of-people-that-can-be-seen-in-a-grid/description/
class Solution:
  def seePeople(self, heights: List[List[int]]) -> List[List[int]]:
    m = len(heights)
    n = len(heights[0])
    ans = [[0] * n for _ in range(m)]

    for i, row in enumerate(heights):
      stack = []
      for j, height in enumerate(row):
        hasEqualHeight = False
        while stack and row[stack[-1]] <= height:
          if row[stack[-1]] == height:
            # edge case: [4, 2, 1, 1, 3]
            hasEqualHeight = True
          ans[i][stack.pop()] += 1
        if stack and not hasEqualHeight:
          ans[i][stack[-1]] += 1
        stack.append(j)

    for j, col in enumerate(zip(*heights)):
      stack = []
      for i, height in enumerate(col):
        hasEqualHeight = False
        while stack and col[stack[-1]] <= height:
          if col[stack[-1]] == height:
            hasEqualHeight = True
          ans[stack.pop()][j] += 1
        if stack and not hasEqualHeight:
          ans[stack[-1]][j] += 1
        stack.append(i)

    return ans


# Link: https://leetcode.com/problems/length-of-the-longest-subsequence-that-sums-to-target/description/
class Solution:
  def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
    # dp[i] := the maximum length of any subsequence of numbers so far that
    # sum to j
    dp = [0] * (target + 1)

    for num in nums:
      for i in range(target, num - 1, -1):
        if i == num or dp[i - num] > 0:
          dp[i] = max(dp[i], 1 + dp[i - num])

    return dp[target] if dp[target] > 0 else -1


# Link: https://leetcode.com/problems/length-of-the-longest-subsequence-that-sums-to-target/description/
class Solution:
  def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
    n = len(nums)
    # dp[i][j] := the maximum length of any subsequence of the first i numbers
    # that sum to j
    dp = [[-1] * (target + 1) for _ in range(n + 1)]

    for i in range(n + 1):
      dp[i][0] = 0

    for i in range(1, n + 1):
      num = nums[i - 1]
      for j in range(1, target + 1):
        # Case 1: Skip `num`.
        if j < num or dp[i - 1][j - num] == -1:
          dp[i][j] = dp[i - 1][j]
        # Case 2: Skip `num` or pick `num`.
        else:
          dp[i][j] = max(dp[i - 1][j], 1 + dp[i - 1][j - num])

    return dp[n][target]


# Link: https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/
class Solution:
  def connect(self, root: 'Node') -> 'Node':
    node = root  # the node that is above the current needling

    while node:
      dummy = Node(0)  # a dummy node before needling
      # Needle the children of the node.
      needle = dummy
      while node:
        if node.left:  # Needle the left child.
          needle.next = node.left
          needle = needle.next
        if node.right:  # Needle the right child.
          needle.next = node.right
          needle = needle.next
        node = node.next
      node = dummy.next  # Move the node to the next level.

    return root


# Link: https://leetcode.com/problems/knight-dialer/description/
class Solution:
  def knightDialer(self, n: int) -> int:
    dirs = ((1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    kMod = 1_000_000_007

    # dp[i][j] := the number of ways stand on (i, j)
    dp = [[1] * 3 for _ in range(4)]
    dp[3][0] = dp[3][2] = 0

    for _ in range(n - 1):
      newDp = [[0] * 3 for _ in range(4)]
      for i in range(4):
        for j in range(3):
          if (i, j) in ((3, 0), (3, 2)):
            continue
          for dx, dy in dirs:
            x = i + dx
            y = j + dy
            if x < 0 or x >= 4 or y < 0 or y >= 3:
              continue
            if (x, y) in ((3, 0), (3, 2)):
              continue
            newDp[x][y] = (newDp[x][y] + dp[i][j]) % kMod
      dp = newDp

    return sum(map(sum, dp)) % kMod


# Link: https://leetcode.com/problems/find-kth-bit-in-nth-binary-string/description/
class Solution:
  def findKthBit(self, n: int, k: int) -> str:
    if n == 1:
      return '0'
    midIndex = pow(2, n - 1)  # 1-indexed
    if k == midIndex:
      return '1'
    if k < midIndex:
      return self.findKthBit(n - 1, k)
    return '1' if self.findKthBit(n - 1, midIndex * 2 - k) == '0' else '0'


# Link: https://leetcode.com/problems/toss-strange-coins/description/
class Solution:
  def probabilityOfHeads(self, prob: List[float], target: int) -> float:
    # dp[j] := the probability of tossing the coins so far with j heads
    dp = [1.0] + [0] * len(prob)

    for p in prob:
      for j in range(target, -1, -1):
        dp[j] = (dp[j - 1] * p if j > 0 else 0) + dp[j] * (1 - p)

    return dp[target]


# Link: https://leetcode.com/problems/toss-strange-coins/description/
class Solution:
  def probabilityOfHeads(self, prob: List[float], target: int) -> float:
    # dp[i][j] := the probability of tossing the first i coins with j heads
    dp = [[0] * (target + 1) for _ in range(len(prob) + 1)]
    dp[0][0] = 1.0

    for i in range(1, len(prob) + 1):
      for j in range(target + 1):
        dp[i][j] = (dp[i - 1][j - 1] * prob[i - 1] if j > 0 else 0) + \
            dp[i - 1][j] * (1 - prob[i - 1])

    return dp[len(prob)][target]


# Link: https://leetcode.com/problems/output-contest-matches/description/
class Solution:
  def findContestMatch(self, n: int) -> str:
    def generateMatches(matches: List[str]) -> str:
      if len(matches) == 1:
        return matches[0]

      nextMatches = []

      for i in range(len(matches) // 2):
        nextMatches.append(
            '(' + matches[i] + ',' + matches[len(matches) - 1 - i] + ')')

      return generateMatches(nextMatches)

    return generateMatches([str(i + 1) for i in range(n)])


# Link: https://leetcode.com/problems/output-contest-matches/description/
class Solution:
  def findContestMatch(self, n: int) -> str:
    matches = [str(i + 1) for i in range(n)]

    while n > 1:
      for i in range(n // 2):
        matches[i] = '(' + matches[i] + ',' + matches[n - 1 - i] + ')'
      n //= 2

    return matches[0]


# Link: https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/
class Solution:
  def longestSubarray(self, nums: List[int]) -> int:
    ans = 0
    count0 = 0

    l = 0
    for r, num in enumerate(nums):
      if num == 0:
        count0 += 1
      while count0 == 2:
        if nums[l] == 0:
          count0 -= 1
        l += 1
      ans = max(ans, r - l)

    return ans


# Link: https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/
class Solution:
  def longestSubarray(self, nums: List[int]) -> int:
    l = 0
    count0 = 0

    for num in nums:
      if num == 0:
        count0 += 1
      if count0 > 1:
        if nums[l] == 0:
          count0 -= 1
        l += 1

    return len(nums) - l - 1


# Link: https://leetcode.com/problems/merge-intervals/description/
class Solution:
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    ans = []

    for interval in sorted(intervals):
      if not ans or ans[-1][1] < interval[0]:
        ans.append(interval)
      else:
        ans[-1][1] = max(ans[-1][1], interval[1])

    return ans


# Link: https://leetcode.com/problems/find-and-replace-in-string/description/
class Solution:
  def findReplaceString(self, s: str, indexes: List[int],
                        sources: List[str], targets: List[str]) -> str:
    for index, source, target in sorted(zip(indexes, sources, targets), reverse=True):
      if s[index:index + len(source)] == source:
        s = s[:index] + target + s[index + len(source):]
    return s


# Link: https://leetcode.com/problems/house-robber/description/
class Solution:
  def rob(self, nums: List[int]) -> int:
    prev1 = 0  # dp[i - 1]
    prev2 = 0  # dp[i - 2]

    for num in nums:
      dp = max(prev1, prev2 + num)
      prev2 = prev1
      prev1 = dp

    return prev1


# Link: https://leetcode.com/problems/house-robber/description/
class Solution:
  def rob(self, nums: List[int]) -> int:
    if not nums:
      return 0
    if len(nums) == 1:
      return nums[0]

    # dp[i]:= max money of robbing nums[0..i]
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
      dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[-1]


# Link: https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/description/
class Solution:
  def minFlips(self, a: int, b: int, c: int) -> int:
    kMaxBit = 30
    ans = 0

    for i in range(kMaxBit):
      if c >> i & 1:
        ans += (a >> i & 1) == 0 and (b >> i & 1) == 0
      else:  # (c >> i & 1) == 0
        ans += (a >> i & 1) + (b >> i & 1)

    return ans


# Link: https://leetcode.com/problems/snakes-and-ladders/description/
class Solution:
  def snakesAndLadders(self, board: List[List[int]]) -> int:
    n = len(board)
    ans = 0
    q = collections.deque([1])
    seen = set()
    A = [0] * (1 + n * n)  # 2D -> 1D

    for i in range(n):
      for j in range(n):
        A[(n - 1 - i) * n + (j + 1 if n - i & 1 else n - j)] = board[i][j]

    while q:
      ans += 1
      for _ in range(len(q)):
        curr = q.popleft()
        for next in range(curr + 1, min(curr + 6, n * n) + 1):
          dest = A[next] if A[next] > 0 else next
          if dest == n * n:
            return ans
          if dest in seen:
            continue
          q.append(dest)
          seen.add(dest)

    return -1


# Link: https://leetcode.com/problems/maximum-subarray-sum-after-one-operation/description/
class Solution:
  def maxSumAfterOperation(self, nums: List[int]) -> int:
    ans = -math.inf
    regular = 0
    squared = 0

    for num in nums:
      squared = max(num**2, regular + num**2, squared + num)
      regular = max(num, regular + num)
      ans = max(ans, squared)

    return ans


# Link: https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors/description/
class Solution:
  def rearrangeArray(self, nums: List[int]) -> List[int]:
    nums.sort()
    for i in range(1, len(nums), 2):
      nums[i], nums[i - 1] = nums[i - 1], nums[i]
    return nums


# Link: https://leetcode.com/problems/reverse-integer/description/
class Solution:
  def reverse(self, x: int) -> int:
    ans = 0
    sign = -1 if x < 0 else 1
    x *= sign

    while x:
      ans = ans * 10 + x % 10
      x //= 10

    return 0 if ans < -2**31 or ans > 2**31 - 1 else sign * ans


# Link: https://leetcode.com/problems/maximal-square/description/
class Solution:
  def maximalSquare(self, matrix: List[List[str]]) -> int:
    m = len(matrix)
    n = len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    maxLength = 0

    for i in range(m):
      for j in range(n):
        if i == 0 or j == 0 or matrix[i][j] == '0':
          dp[i][j] = 1 if matrix[i][j] == '1' else 0
        else:
          dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1]
                         [j], dp[i][j - 1]) + 1
        maxLength = max(maxLength, dp[i][j])

    return maxLength * maxLength


# Link: https://leetcode.com/problems/maximal-square/description/
class Solution:
  def maximalSquare(self, matrix: List[List[chr]]) -> int:
    m = len(matrix)
    n = len(matrix[0])
    dp = [0] * n
    maxLength = 0
    prev = 0  # dp[i - 1][j - 1]

    for i in range(m):
      for j in range(n):
        cache = dp[j]
        if i == 0 or j == 0 or matrix[i][j] == '0':
          dp[j] = 1 if matrix[i][j] == '1' else 0
        else:
          dp[j] = min([prev, dp[j], dp[j - 1]]) + 1
        maxLength = max(maxLength, dp[j])
        prev = cache

    return maxLength * maxLength


# Link: https://leetcode.com/problems/number-of-distinct-islands/description/
class Solution:
  def numDistinctIslands(self, grid: List[List[int]]) -> int:
    seen = set()

    def dfs(i: int, j: int, i0: int, j0: int):
      if i < 0 or i == len(grid) or j < 0 or j == len(grid[0]):
        return
      if grid[i][j] == 0 or (i, j) in seen:
        return

      seen.add((i, j))
      island.append((i - i0, j - j0))
      dfs(i + 1, j, i0, j0)
      dfs(i - 1, j, i0, j0)
      dfs(i, j + 1, i0, j0)
      dfs(i, j - 1, i0, j0)

    islands = set()  # all the different islands

    for i in range(len(grid)):
      for j in range(len(grid[0])):
        island = []
        dfs(i, j, i, j)
        if island:
          islands.add(frozenset(island))

    return len(islands)


# Link: https://leetcode.com/problems/longest-line-of-consecutive-one-in-matrix/description/
class Solution:
  def longestLine(self, mat: List[List[int]]) -> int:
    m = len(mat)
    n = len(mat[0])
    ans = 0
    # dp[i][j][0] := horizontal
    # dp[i][j][1] := vertical
    # dp[i][j][2] := diagonal
    # dp[i][j][3] := anti-diagonal
    dp = [[[0] * 4 for j in range(n)] for _ in range(m)]

    for i in range(m):
      for j in range(n):
        if mat[i][j] == 1:
          dp[i][j][0] = dp[i][j - 1][0] + 1 if j > 0 else 1
          dp[i][j][1] = dp[i - 1][j][1] + 1 if i > 0 else 1
          dp[i][j][2] = dp[i - 1][j - 1][2] + 1 if i > 0 and j > 0 else 1
          dp[i][j][3] = dp[i - 1][j + 1][3] + 1 if i > 0 and j < n - 1 else 1
          ans = max(ans, max(dp[i][j]))

    return ans


# Link: https://leetcode.com/problems/find-positive-integer-solution-for-a-given-equation/description/
class Solution:
  def findSolution(self, customfunction: 'CustomFunction', z: int) -> List[List[int]]:
    ans = []
    x = 1
    y = 1000

    while x <= 1000 and y >= 1:
      f = customfunction.f(x, y)
      if f < z:
        x += 1
      elif f > z:
        y -= 1
      else:
        ans.append([x, y])
        x += 1
        y -= 1

    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-reduce-an-integer-to-0/description/
class Solution:
  def minOperations(self, n: int) -> int:
    # The strategy is that when the end of n is
    #   1. consecutive 1s, add 1 (2^0).
    #   2. single 1, substract 1 (2^0).
    #   3. 0, substract 2^k to omit the last 1. Equivalently, n >> 1.
    #
    # e.g.
    #
    #         n = 0b101
    # n -= 2^0 -> 0b100
    # n -= 2^2 -> 0b0
    #         n = 0b1011
    # n += 2^0 -> 0b1100
    # n -= 2^2 -> 0b1000
    # n -= 2^3 -> 0b0
    ans = 0

    while n > 0:
      if (n & 3) == 3:
        n += 1
        ans += 1
      elif (n & 1) == 1:
        n -= 1
        ans += 1
      else:
        n >>= 1

    return ans


# Link: https://leetcode.com/problems/count-sub-islands/description/
class Solution:
  def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
    m = len(grid2)
    n = len(grid2[0])

    def dfs(i: int, j: int) -> int:
      if i < 0 or i == m or j < 0 or j == n:
        return 1
      if grid2[i][j] != 1:
        return 1

      grid2[i][j] = 2  # Mark 2 as visited.

      return dfs(i + 1, j) & dfs(i - 1, j) & \
          dfs(i, j + 1) & dfs(i, j - 1) & grid1[i][j]

    ans = 0

    for i in range(m):
      for j in range(n):
        if grid2[i][j] == 1:
          ans += dfs(i, j)

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-ii/description/
class Solution:
  def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
    bannedSet = set(banned)
    l = 1
    r = n

    while l < r:
      m = (l + r + 1) // 2
      if self._getSum(bannedSet, m) > maxSum:
        r = m - 1
      else:
        l = m

    return l - sum(b <= l for b in banned)

  # Returns sum([1..m]) - sum(bannedSet).
  def _getSum(self, bannedSet: Set[int], m: int) -> int:
    return m * (m + 1) // 2 - sum(b for b in bannedSet if b <= m)


# Link: https://leetcode.com/problems/number-of-subarrays-with-gcd-equal-to-k/description/
class Solution:
  def subarrayGCD(self, nums: List[int], k: int) -> int:
    ans = 0
    gcds = collections.Counter()

    for num in nums:
      if num % k == 0:
        nextGcds = collections.defaultdict(int)
        nextGcds[num] += 1
        for prevGcd, count in gcds.items():
          nextGcds[math.gcd(prevGcd, num)] += count
        ans += nextGcds.get(k, 0)
        gcds = nextGcds
      else:
        # The GCD streak stops, so fresh start from the next number.
        gcds.clear()

    return ans


# Link: https://leetcode.com/problems/product-of-array-except-self/description/
class Solution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = [1] * n

    # Use ans as the prefix product array.
    for i in range(1, n):
      ans[i] = ans[i - 1] * nums[i - 1]

    suffix = 1  # suffix product
    for i, num in reversed(list(enumerate(nums))):
      ans[i] *= suffix
      suffix *= num

    return ans


# Link: https://leetcode.com/problems/product-of-array-except-self/description/
class Solution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:
    n = len(nums)
    prefix = [1] * n  # prefix product
    suffix = [1] * n  # suffix product

    for i in range(1, n):
      prefix[i] = prefix[i - 1] * nums[i - 1]

    for i in reversed(range(n - 1)):
      suffix[i] = suffix[i + 1] * nums[i + 1]

    return [prefix[i] * suffix[i] for i in range(n)]


# Link: https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/description/
class Solution:
  def getMinSwaps(self, num: str, k: int) -> int:
    def nextPermutation(nums: List[int]):
      n = len(nums)

      # From the back to the front, find the first num < nums[i + 1].
      i = n - 2
      while i >= 0:
        if nums[i] < nums[i + 1]:
          break
        i -= 1

      # From the back to the front, find the first num > nums[i] and swap it with nums[i].
      if i >= 0:
        for j in range(n - 1, i, -1):
          if nums[j] > nums[i]:
            nums[i], nums[j] = nums[j], nums[i]
            break

      def reverse(nums, l, r):
        while l < r:
          nums[l], nums[r] = nums[r], nums[l]
          l += 1
          r -= 1

      # Reverse nums[i + 1..n - 1]
      reverse(nums, i + 1, len(nums) - 1)

    A = [int(c) for c in num]  # Original
    B = A.copy()  # Permutated

    for _ in range(k):
      nextPermutation(B)

    def countSteps(A: List[int], B: List[int]) -> int:
      count = 0

      j = 0
      for i in range(len(A)):
        j = i
        while A[i] != B[j]:
          j += 1
        while i < j:
          B[j], B[j - 1] = B[j - 1], B[j]
          j -= 1
          count += 1

      return count

    return countSteps(A, B)


# Link: https://leetcode.com/problems/make-the-prefix-sum-non-negative/description/
class Solution:
  def makePrefSumNonNegative(self, nums: List[int]) -> int:
    ans = 0
    prefix = 0
    minHeap = []

    for num in nums:
      prefix += num
      if num < 0:
        heapq.heappush(minHeap, num)
      while prefix < 0:
        prefix -= heapq.heappop(minHeap)
        ans += 1

    return ans


# Link: https://leetcode.com/problems/squirrel-simulation/description/
class Solution:
  def minDistance(self, height: int, width: int, tree: List[int], squirrel: List[int], nuts: List[List[int]]) -> int:
    def dist(a: List[int], b: List[int]) -> int:
      return abs(a[0] - b[0]) + abs(a[1] - b[1])

    totDist = sum(dist(nut, tree) for nut in nuts) * 2
    maxSave = max(dist(nut, tree) - dist(nut, squirrel) for nut in nuts)
    return totDist - maxSave


# Link: https://leetcode.com/problems/number-of-closed-islands/description/
class Solution:
  def closedIsland(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    def dfs(i: int, j: int) -> None:
      if i < 0 or i == m or j < 0 or j == n:
        return
      if grid[i][j] == 1:
        return

      grid[i][j] = 1
      dfs(i + 1, j)
      dfs(i - 1, j)
      dfs(i, j + 1)
      dfs(i, j - 1)

    # Remove the lands connected to the edge.
    for i in range(m):
      for j in range(n):
        if i * j == 0 or i == m - 1 or j == n - 1:
          if grid[i][j] == 0:
            dfs(i, j)

    ans = 0

    # Reduce to 200. Number of Islands
    for i in range(m):
      for j in range(n):
        if grid[i][j] == 0:
          dfs(i, j)
          ans += 1

    return ans


# Link: https://leetcode.com/problems/coloring-a-border/description/
class Solution:
  def colorBorder(self, grid: List[List[int]], r0: int, c0: int, color: int) -> List[List[int]]:
    def dfs(i: int, j: int, originalColor: int) -> None:
      if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] != originalColor:
        return

      grid[i][j] = -originalColor
      dfs(i + 1, j, originalColor)
      dfs(i - 1, j, originalColor)
      dfs(i, j + 1, originalColor)
      dfs(i, j - 1, originalColor)

      if 0 < i < len(grid) - 1 and 0 < j < len(grid[0]) - 1 and \
              abs(grid[i + 1][j]) == originalColor and \
              abs(grid[i - 1][j]) == originalColor and \
              abs(grid[i][j + 1]) == originalColor and \
              abs(grid[i][j - 1]) == originalColor:
        grid[i][j] = originalColor

    dfs(r0, c0, grid[r0][c0])

    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if grid[i][j] < 0:
          grid[i][j] = color

    return grid


# Link: https://leetcode.com/problems/find-the-index-of-the-large-integer/description/
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
# class ArrayReader(object):
# # Compares the sum of arr[l..r] with the sum of arr[x..y]
# # return 1 if sum(arr[l..r]) > sum(arr[x..y])
# # return 0 if sum(arr[l..r]) == sum(arr[x..y])
# # return -1 if sum(arr[l..r]) < sum(arr[x..y])
#   def compareSub(self, l: int, r: int, x: int, y: int) -> int:
#
# # Returns the length of the array
#   def length(self) -> int:
#


class Solution:
  def getIndex(self, reader: 'ArrayReader') -> int:
    l = 0
    r = reader.length() - 1

    while l < r:
      m = (l + r) // 2
      res = reader.compareSub(l, m, m + 1, r) if (r - l + 1) % 2 == 0 \
          else reader.compareSub(l, m, m, r)
      if res == -1:
        l = m + 1
      else:  # res == 1 or res == 0
        r = m

    return l


# Link: https://leetcode.com/problems/find-the-index-of-the-large-integer/description/
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
# class ArrayReader(object):
# # Compares the sum of arr[l..r] with the sum of arr[x..y]
# # return 1 if sum(arr[l..r]) > sum(arr[x..y])
# # return 0 if sum(arr[l..r]) == sum(arr[x..y])
# # return -1 if sum(arr[l..r]) < sum(arr[x..y])
#   def compareSub(self, l: int, r: int, x: int, y: int) -> int:
#
# # Returns the length of the array
#   def length(self) -> int:
#


class Solution:
  def getIndex(self, reader: 'ArrayReader') -> int:
    l = 0
    r = reader.length() - 1

    while l < r:
      m = (l + r) // 2
      if (r - l) % 2 == 0:
        res = reader.compareSub(l, m - 1, m + 1, r)
        if res == 0:
          return m
        if res == 1:
          r = m - 1
        else:  # res == -1
          l = m + 1
      else:
        res = reader.compareSub(l, m, m + 1, r)
        # res is either 1 or -1.
        if res == 1:
          r = m
        else:  # res == -1
          l = m + 1

    return l


# Link: https://leetcode.com/problems/maximum-profit-from-trading-stocks/description/
class Solution:
  def maximumProfit(self, present: List[int], future: List[int], budget: int) -> int:
    n = len(present)
    # dp[i] := the maximum profit of buying present so far with i budget
    dp = [0] * (budget + 1)

    for p, f in zip(present, future):
      for j in range(budget, p - 1, -1):
        dp[j] = max(dp[j], f - p + dp[j - p])

    return dp[budget]


# Link: https://leetcode.com/problems/maximum-profit-from-trading-stocks/description/
class Solution:
  def maximumProfit(self, present: List[int], future: List[int], budget: int) -> int:
    n = len(present)
    # dp[i][j] := the maximum profit of buying present[0..i) with j budget
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
      profit = future[i - 1] - present[i - 1]
      for j in range(budget + 1):
        if j < present[i - 1]:
          dp[i][j] = dp[i - 1][j]
        else:
          dp[i][j] = max(dp[i - 1][j], profit + dp[i - 1][j - present[i - 1]])

    return dp[n][budget]


# Link: https://leetcode.com/problems/ambiguous-coordinates/description/
class Solution:
  def ambiguousCoordinates(self, s: str) -> List[str]:
    def splits(s: str) -> List[str]:
      if not s or len(s) > 1 and s[0] == s[-1] == '0':
        return []
      if s[-1] == '0':
        return [s]
      if s[0] == '0':
        return [s[0] + '.' + s[1:]]
      return [s] + [s[:i] + '.' + s[i:] for i in range(1, len(s))]

    ans = []
    s = s[1:-1]

    for i in range(1, len(s)):
      for x in splits(s[:i]):
        for y in splits(s[i:]):
          ans.append('(%s, %s)' % (x, y))

    return ans


# Link: https://leetcode.com/problems/merge-triplets-to-form-target-triplet/description/
class Solution:
  def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
    merged = [0] * len(target)

    for triplet in triplets:
      if all(a <= b for a, b in zip(triplet, target)):
        for i in range(3):
          merged[i] = max(merged[i], triplet[i])

    return merged == target


# Link: https://leetcode.com/problems/moving-stones-until-consecutive/description/
class Solution:
  def numMovesStones(self, a: int, b: int, c: int) -> List[int]:
    nums = sorted([a, b, c])

    if nums[2] - nums[0] == 2:
      return [0, 0]
    return [1 if min(nums[1] - nums[0], nums[2] - nums[1]) <= 2 else 2,
            nums[2] - nums[0] - 2]


# Link: https://leetcode.com/problems/subsequence-of-size-k-with-the-largest-even-sum/description/
class Solution:
  def largestEvenSum(self, nums: List[int], k: int) -> int:
    nums.sort()
    summ = sum(nums[-k:])
    if summ % 2 == 0:
      return summ

    minOdd = -1
    minEven = -1
    maxOdd = -1
    maxEven = -1

    for i in range(len(nums) - 1, len(nums) - k - 1, -1):
      if nums[i] & 1:
        minOdd = nums[i]
      else:
        minEven = nums[i]

    for i in range(len(nums) - k):
      if nums[i] & 1:
        maxOdd = nums[i]
      else:
        maxEven = nums[i]

    ans = -1

    if maxEven >= 0 and minOdd >= 0:
      ans = max(ans, summ + maxEven - minOdd)
    if maxOdd >= 0 and minEven >= 0:
      ans = max(ans, summ + maxOdd - minEven)
    return ans


# Link: https://leetcode.com/problems/number-of-distinct-binary-strings-after-applying-operations/description/
class Solution:
  def countDistinctStrings(self, s: str, k: int) -> int:
    # Since the content of `s` doesn't matter, for each i in [0, n - k], we can
    # flip s[i..i + k] or don't flip it. Therefore, there's 2^(n - k + 1) ways.
    return pow(2, len(s) - k + 1, 1_000_000_007)


# Link: https://leetcode.com/problems/number-of-ways-to-split-a-string/description/
class Solution:
  def numWays(self, s: str) -> int:
    kMod = 1_000_000_007
    ones = s.count('1')
    if ones % 3 != 0:
      return 0
    if ones == 0:
      n = len(s)
      return (n - 1) * (n - 2) // 2 % kMod

    s1End = -1
    s2Start = -1
    s2End = -1
    s3Start = -1
    onesSoFar = 0

    for i, c in enumerate(s):
      if c == '1':
        onesSoFar += 1
      if s1End == -1 and onesSoFar == ones // 3:
        s1End = i
      elif s2Start == -1 and onesSoFar == ones // 3 + 1:
        s2Start = i
      if s2End == -1 and onesSoFar == ones // 3 * 2:
        s2End = i
      elif s3Start == -1 and onesSoFar == ones // 3 * 2 + 1:
        s3Start = i

    return (s2Start - s1End) * (s3Start - s2End) % kMod


# Link: https://leetcode.com/problems/most-expensive-item-that-can-not-be-bought/description/
class Solution:
  def mostExpensiveItem(self, primeOne: int, primeTwo: int) -> int:
    # https://en.wikipedia.org/wiki/Coin_problem
    return primeOne * primeTwo - primeOne - primeTwo


# Link: https://leetcode.com/problems/utf-8-validation/description/
class Solution:
  def validUtf8(self, data: List[int]) -> bool:
    followedBytes = 0

    for d in data:
      if followedBytes == 0:
        if (d >> 3) == 0b11110:
          followedBytes = 3
        elif (d >> 4) == 0b1110:
          followedBytes = 2
        elif (d >> 5) == 0b110:
          followedBytes = 1
        elif (d >> 7) == 0b0:
          followedBytes = 0
        else:
          return False
      else:
        if (d >> 6) != 0b10:
          return False
        followedBytes -= 1

    return followedBytes == 0


# Link: https://leetcode.com/problems/longest-repeating-substring/description/
class Solution:
  def longestRepeatingSubstring(self, s: str) -> int:
    n = len(s)
    ans = 0
    # dp[i][j] := the number of repeating characters of s[0..i) and s[0..j)
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
      for j in range(i + 1, n + 1):
        if s[i - 1] == s[j - 1]:
          dp[i][j] = 1 + dp[i - 1][j - 1]
          ans = max(ans, dp[i][j])

    return ans


# Link: https://leetcode.com/problems/design-add-and-search-words-data-structure/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.isWord = False


class WordDictionary:
  def __init__(self):
    self.root = TrieNode()

  def addWord(self, word: str) -> None:
    node: TrieNode = self.root
    for c in word:
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True

  def search(self, word: str) -> bool:
    return self._dfs(word, 0, self.root)

  def _dfs(self, word: str, s: int, node: TrieNode) -> bool:
    if s == len(word):
      return node.isWord
    if word[s] != '.':
      child: TrieNode = node.children.get(word[s], None)
      return self._dfs(word, s + 1, child) if child else False
    return any(self._dfs(word, s + 1, child) for child in node.children.values())


# Link: https://leetcode.com/problems/maximum-subarray-min-product/description/
class Solution:
  def maxSumMinProduct(self, nums: List[int]) -> int:
    ans = 0
    stack = []
    prefix = [0] + list(itertools.accumulate(nums))

    for i in range(len(nums) + 1):
      while stack and (i == len(nums) or nums[stack[-1]] > nums[i]):
        minVal = nums[stack.pop()]
        summ = prefix[i] - prefix[stack[-1] + 1] if stack else prefix[i]
        ans = max(ans, minVal * summ)
      stack.append(i)

    return ans % int(1e9 + 7)


# Link: https://leetcode.com/problems/brace-expansion/description/
class Solution:
  def expand(self, s: str) -> List[str]:
    ans = []

    def dfs(i: int, path: List[str]) -> None:
      if i == len(s):
        ans.append(''.join(path))
        return
      if s[i] == '{':
        nextRightBraceIndex = s.find('}', i)
        for c in s[i + 1:nextRightBraceIndex].split(','):
          path.append(c)
          dfs(nextRightBraceIndex + 1, path)
          path.pop()
      else:  # s[i] != '{'
        path.append(s[i])
        dfs(i + 1, path)
        path.pop()

    dfs(0, [])
    return sorted(ans)


# Link: https://leetcode.com/problems/spiral-matrix/description/
class Solution:
  def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if not matrix:
      return []

    m = len(matrix)
    n = len(matrix[0])
    ans = []
    r1 = 0
    c1 = 0
    r2 = m - 1
    c2 = n - 1

    # Repeatedly add matrix[r1..r2][c1..c2] to `ans`.
    while len(ans) < m * n:
      j = c1
      while j <= c2 and len(ans) < m * n:
        ans.append(matrix[r1][j])
        j += 1
      i = r1 + 1
      while i <= r2 - 1 and len(ans) < m * n:
        ans.append(matrix[i][c2])
        i += 1
      j = c2
      while j >= c1 and len(ans) < m * n:
        ans.append(matrix[r2][j])
        j -= 1
      i = r2 - 1
      while i >= r1 + 1 and len(ans) < m * n:
        ans.append(matrix[i][c1])
        i -= 1
      r1 += 1
      c1 += 1
      r2 -= 1
      c2 -= 1

    return ans


# Link: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/description/
class Solution:
  def findMaximumXOR(self, nums: List[int]) -> int:
    maxNum = max(nums)
    if maxNum == 0:
      return 0
    maxBit = int(math.log2(maxNum))
    ans = 0
    prefixMask = 0  # `prefixMask` grows like: 10000 -> 11000 -> ... -> 11111.

    # If ans is 11100 when i = 2, it means that before we reach the last two
    # bits, 11100 is the maximum XOR we have, and we're going to explore if we
    # can get another two 1s and put them into `ans`.
    for i in range(maxBit, -1, -1):
      prefixMask |= 1 << i
      # We only care about the left parts,
      # If i = 2, nums = [1110, 1011, 0111]
      #    -> prefixes = [1100, 1000, 0100]
      prefixes = set([num & prefixMask for num in nums])
      # If i = 1 and before this iteration, the ans is 10100, it means that we
      # want to grow the ans to 10100 | 1 << 1 = 10110 and we're looking for
      # XOR of two prefixes = candidate.
      candidate = ans | 1 << i
      for prefix in prefixes:
        if prefix ^ candidate in prefixes:
          ans = candidate
          break

    return ans


# Link: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/description/
class TrieNode:
  def __init__(self):
    self.children: List[Optional[TrieNode]] = [None] * 2


class BitTrie:
  def __init__(self, maxBit: int):
    self.maxBit = maxBit
    self.root = TrieNode()

  def insert(self, num: int) -> None:
    node = self.root
    for i in range(self.maxBit, -1, -1):
      bit = num >> i & 1
      if not node.children[bit]:
        node.children[bit] = TrieNode()
      node = node.children[bit]

  def getMaxXor(self, num: int) -> int:
    maxXor = 0
    node = self.root
    for i in range(self.maxBit, -1, -1):
      bit = num >> i & 1
      toggleBit = bit ^ 1
      if node.children[toggleBit]:
        maxXor = maxXor | 1 << i
        node = node.children[toggleBit]
      elif node.children[bit]:
        node = node.children[bit]
      else:  # There's nothing in the Bit Trie.
        return 0
    return maxXor


class Solution:
  def findMaximumXOR(self, nums: List[int]) -> int:
    maxNum = max(nums)
    if maxNum == 0:
      return 0
    maxBit = int(math.log2(maxNum))
    ans = 0
    bitTrie = BitTrie(maxBit)

    for num in nums:
      ans = max(ans, bitTrie.getMaxXor(num))
      bitTrie.insert(num)

    return ans


# Link: https://leetcode.com/problems/unique-paths/description/
class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    dp = [1] * n

    for _ in range(1, m):
      for j in range(1, n):
        dp[j] += dp[j - 1]

    return dp[n - 1]


# Link: https://leetcode.com/problems/unique-paths/description/
class Solution:
  def uniquePaths(self, m: int, n: int) -> int:
    # dp[i][j] := the number of unique paths from (0, 0) to (i, j)
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
      for j in range(1, n):
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[-1][-1]


# Link: https://leetcode.com/problems/plates-between-candles/description/
class Solution:
  def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
    n = len(s)
    ans = []
    closestLeftCandle = [0] * n
    closestRightCandle = [0] * n
    candleCount = [0] * n  # candleCount[i] := the number of candles in s[0..i]
    candle = -1
    count = 0

    for i, c in enumerate(s):
      if c == '|':
        candle = i
        count += 1
      closestLeftCandle[i] = candle
      candleCount[i] = count

    candle = -1
    for i, c in reversed(list(enumerate(s))):
      if c == '|':
        candle = i
      closestRightCandle[i] = candle

    for left, right in queries:
      l = closestRightCandle[left]
      r = closestLeftCandle[right]
      if l == -1 or r == -1 or l > r:
        ans.append(0)
      else:
        lengthBetweenCandles = r - l + 1
        numCandles = candleCount[r] - candleCount[l] + 1
        ans.append(lengthBetweenCandles - numCandles)

    return ans


# Link: https://leetcode.com/problems/plates-between-candles/description/
class Solution:
  def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
    ans = []
    indices = [i for i, c in enumerate(s) if c == '|']  # indices of '|'

    for left, right in queries:
      l = bisect.bisect_left(indices, left)
      r = bisect.bisect_right(indices, right) - 1
      if l < r:
        lengthBetweenCandles = indices[r] - indices[l] + 1
        numCandles = r - l + 1
        ans.append(lengthBetweenCandles - numCandles)
      else:
        ans.append(0)

    return ans


# Link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/
class Solution:
  def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    slow = head
    fast = head

    for _ in range(n):
      fast = fast.next
    if not fast:
      return head.next

    while fast.next:
      slow = slow.next
      fast = fast.next
    slow.next = slow.next.next

    return head


# Link: https://leetcode.com/problems/largest-time-for-given-digits/description/
class Solution:
  def largestTimeFromDigits(self, arr: List[int]) -> str:
    for time in itertools.permutations(sorted(arr, reverse=True)):
      if time[:2] < (2, 4) and time[2] < 6:
        return '%d%d:%d%d' % time
    return ''


# Link: https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/description/
class Solution:
  # Same as 926. Flip String to Monotone Increasing
  def minimumDeletions(self, s: str) -> int:
    dp = 0  # the number of characters to be deleted to make subso far balanced
    countB = 0

    for c in s:
      if c == 'a':
        # 1. Delete 'a'.
        # 2. Keep 'a' and delete the previous 'b's.
        dp = min(dp + 1, countB)
      else:
        countB += 1

    return dp


# Link: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/
class Solution:
  def flatten(self, root: Optional[TreeNode]) -> None:
    if not root:
      return

    self.flatten(root.left)
    self.flatten(root.right)

    left = root.left  # flattened left
    right = root.right  # flattened right

    root.left = None
    root.right = left

    # Connect the original right subtree to the end of the new right subtree.
    rightmost = root
    while rightmost.right:
      rightmost = rightmost.right
    rightmost.right = right


# Link: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/
class Solution:
  def flatten(self, root: Optional[TreeNode]) -> None:
    if not root:
      return

    while root:
      if root.left:
        # Find the rightmost root
        rightmost = root.left
        while rightmost.right:
          rightmost = rightmost.right
        # Rewire the connections
        rightmost.right = root.right
        root.right = root.left
        root.left = None
      # Move on to the right side of the tree
      root = root.right


# Link: https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/
class Solution:
  def flatten(self, root: Optional[TreeNode]) -> None:
    if not root:
      return

    stack = [root]

    while stack:
      root = stack.pop()
      if root.right:
        stack.append(root.right)
      if root.left:
        stack.append(root.left)
      if stack:
        root.right = stack[-1]
      root.left = None


# Link: https://leetcode.com/problems/rle-iterator/description/
class RLEIterator:
  def __init__(self, encoding: List[int]):
    self.encoding = encoding
    self.index = 0

  def next(self, n: int) -> int:
    while self.index < len(self.encoding) and self.encoding[self.index] < n:
      n -= self.encoding[self.index]
      self.index += 2

    if self.index == len(self.encoding):
      return -1

    self.encoding[self.index] -= n
    return self.encoding[self.index + 1]


# Link: https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/description/
class Solution:
  def validPartition(self, nums: List[int]) -> bool:
    n = len(nums)
    # dp[i] := True if there's a valid partition for the first i numbers
    dp = [False] * (n + 1)
    dp[0] = True
    dp[2] = nums[0] == nums[1]

    for i in range(3, n + 1):
      dp[i] = (dp[i - 2] and nums[i - 2] == nums[i - 1]) or \
          (dp[i - 3] and ((nums[i - 3] == nums[i - 2] and nums[i - 2] == nums[i - 1]) or
                          (nums[i - 3] + 1 == nums[i - 2] and nums[i - 2] + 1 == nums[i - 1])))

    return dp[n]


# Link: https://leetcode.com/problems/lru-cache/description/
class Node:
  def __init__(self, key: int, value: int):
    self.key = key
    self.value = value
    self.prev = None
    self.next = None


class LRUCache:
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.keyToNode = {}
    self.head = Node(-1, -1)
    self.tail = Node(-1, -1)
    self.join(self.head, self.tail)

  def get(self, key: int) -> int:
    if key not in self.keyToNode:
      return -1

    node = self.keyToNode[key]
    self.remove(node)
    self.moveToHead(node)
    return node.value

  def put(self, key: int, value: int) -> None:
    if key in self.keyToNode:
      node = self.keyToNode[key]
      node.value = value
      self.remove(node)
      self.moveToHead(node)
      return

    if len(self.keyToNode) == self.capacity:
      lastNode = self.tail.prev
      del self.keyToNode[lastNode.key]
      self.remove(lastNode)

    self.moveToHead(Node(key, value))
    self.keyToNode[key] = self.head.next

  def join(self, node1: Node, node2: Node):
    node1.next = node2
    node2.prev = node1

  def moveToHead(self, node: Node):
    self.join(node, self.head.next)
    self.join(self.head, node)

  def remove(self, node: Node):
    self.join(node.prev, node.next)


# Link: https://leetcode.com/problems/first-completely-painted-row-or-column/description/
class Solution:
  def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
    m = len(mat)
    n = len(mat[0])
    # rows[i] := the number of painted grid in the i-th row
    rows = [0] * m
    # cols[j] := the number of painted grid in the j-th column
    cols = [0] * n
    # numToRow[num] := the i-th row of `num` in `mat`
    numToRow = [0] * (m * n + 1)
    # numToCol[num] := the j-th column of `num` in `mat`
    numToCol = [0] * (m * n + 1)

    for i, row in enumerate(mat):
      for j, num in enumerate(row):
        numToRow[num] = i
        numToCol[num] = j

    for i, a in enumerate(arr):
      rows[numToRow[a]] += 1
      if rows[numToRow[a]] == n:
        return i
      cols[numToCol[a]] += 1
      if cols[numToCol[a]] == m:
        return i


# Link: https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/description/
import statistics


class Solution:
  def minMoves2(self, nums: List[int]) -> int:
    median = int(statistics.median(nums))
    return sum(abs(num - median) for num in nums)


# Link: https://leetcode.com/problems/minimum-garden-perimeter-to-collect-enough-apples/description/
class Solution:
  def minimumPerimeter(self, neededApples: int) -> int:
    def numApples(k: int) -> int:
      """Returns the number of apples at the k-th level.

         k := the level making perimeter = 8k
      p(k) := the number of apples at the k-th level on the perimeter
      n(k) := the number of apples at the k-th level not no the perimeter

      p(1) =             1 + 2
      p(2) =         3 + 2 + 3 + 4
      p(3) =     5 + 4 + 3 + 4 + 5 + 6
      p(4) = 7 + 6 + 5 + 4 + 5 + 6 + 7 + 8
      p(k) = k + 2(k+1) + 2(k+2) + ... + 2(k+k-1) + 2k
          = k + 2k^2 + 2*k(k-1)//2
          = k + 2k^2 + k^2 - k = 3k^2

      n(k) = p(1) + p(2) + p(3) + ... + p(k)
          = 3*1  + 3*4  + 3*9  + ... + 3*k^2
          = 3 * (1 + 4 + 9 + ... + k^2)
          = 3 * k(k+1)(2k+1)//6 = k(k+1)(2k+1)//2
      So, the number of apples at the k-th level should be
        k(k+1)(2k+1)//2 * 4 = 2k(k+1)(2k+1)
      """
      return 2 * k * (k + 1) * (2 * k + 1)

    return bisect.bisect_left(range(100_000), neededApples,
                              key=lambda m: numApples(m)) * 8


# Link: https://leetcode.com/problems/sort-vowels-in-a-string/description/
class Solution:
  def sortVowels(self, s: str) -> str:
    kVowels = 'aeiouAEIOU'
    ans = []
    vowels = sorted([c for c in s if c in kVowels])

    i = 0  # vowels' index
    for c in s:
      if c in kVowels:
        ans.append(vowels[i])
        i += 1
      else:
        ans.append(c)

    return ''.join(ans)


# Link: https://leetcode.com/problems/number-of-subarrays-having-even-product/description/
class Solution:
  def evenProduct(self, nums: List[int]) -> int:
    ans = 0
    numsBeforeEven = 0  # inclusively

    # e.g. nums = [1, 0, 1, 1, 0].
    # After meeting the first 0, set `numsBeforeEven` to 2. So, the number
    # between index 1 to index 3 (the one before next 0) will contribute 2 to
    # `ans`.
    for i, num in enumerate(nums):
      if num % 2 == 0:
        numsBeforeEven = i + 1
      ans += numsBeforeEven

    return ans


# Link: https://leetcode.com/problems/largest-1-bordered-square/description/
class Solution:
  def largest1BorderedSquare(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    # leftOnes[i][j] := consecutive 1s in the left of grid[i][j]
    leftOnes = [[0] * n for _ in range(m)]
    # topOnes[i][j] := consecutive 1s in the top of grid[i][j]
    topOnes = [[0] * n for _ in range(m)]

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1:
          leftOnes[i][j] = 1 if j == 0 else 1 + leftOnes[i][j - 1]
          topOnes[i][j] = 1 if i == 0 else 1 + topOnes[i - 1][j]

    for sz in range(min(m, n), 0, -1):
      for i in range(m - sz + 1):
        for j in range(n - sz + 1):
          x = i + sz - 1
          y = j + sz - 1
          # If grid[i..x][j..y] has all 1s on its border.
          if min(leftOnes[i][y], leftOnes[x][y], topOnes[x][j], topOnes[x][y]) >= sz:
            return sz * sz

    return 0


# Link: https://leetcode.com/problems/design-a-stack-with-increment-operation/description/
class CustomStack:
  def __init__(self, maxSize: int):
    self.maxSize = maxSize
    self.stack = []
    # pendingIncrements[i] := the pending increment for stack[0..i].
    self.pendingIncrements = []

  def push(self, x: int) -> None:
    if len(self.stack) == self.maxSize:
      return
    self.stack.append(x)
    self.pendingIncrements.append(0)

  def pop(self) -> int:
    if not self.stack:
      return -1
    if len(self.stack) > 1:
      self.pendingIncrements[-2] += self.pendingIncrements[-1]
    return self.stack.pop() + self.pendingIncrements.pop()

  def increment(self, k: int, val: int) -> None:
    if not self.stack:
      return
    i = min(k - 1, len(self.stack) - 1)
    self.pendingIncrements[i] += val


# Link: https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/
class Solution:
  def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
    if len(bloomDay) < m * k:
      return -1

    def getBouquetCount(waitingDays: int) -> int:
      """
      Returns the number of bouquets (k flowers needed) can be made after the
      `waitingDays`.
      """
      bouquetCount = 0
      requiredFlowers = k
      for day in bloomDay:
        if day > waitingDays:
          # Reset `requiredFlowers` since there was not enough adjacent flowers.
          requiredFlowers = k
        else:
          requiredFlowers -= 1
          if requiredFlowers == 0:
            # Use k adjacent flowers to make a bouquet.
            bouquetCount += 1
            requiredFlowers = k
      return bouquetCount

    l = min(bloomDay)
    r = max(bloomDay)

    while l < r:
      mid = (l + r) // 2
      if getBouquetCount(mid) >= m:
        r = mid
      else:
        l = mid + 1

    return l


# Link: https://leetcode.com/problems/count-numbers-with-unique-digits/description/
class Solution:
  def countNumbersWithUniqueDigits(self, n: int) -> int:
    if n == 0:
      return 1

    ans = 10
    uniqueDigits = 9
    availableNum = 9

    while n > 1 and availableNum > 0:
      uniqueDigits *= availableNum
      ans += uniqueDigits
      n -= 1
      availableNum -= 1

    return ans


# Link: https://leetcode.com/problems/unique-binary-search-trees-ii/description/
class Solution:
  def generateTrees(self, n: int) -> List[TreeNode]:
    if n == 0:
      return []

    def generateTrees(mini: int, maxi: int) -> List[Optional[int]]:
      if mini > maxi:
        return [None]

      ans = []

      for i in range(mini, maxi + 1):
        for left in generateTrees(mini, i - 1):
          for right in generateTrees(i + 1, maxi):
            ans.append(TreeNode(i))
            ans[-1].left = left
            ans[-1].right = right

      return ans

    return generateTrees(1, n)


# Link: https://leetcode.com/problems/linked-list-random-node/description/
# Definition for singly-linked list.
# class ListNode:
#   def __init__(self, val=0, next=None):
#     self.val = val
#     self.next = next

class Solution:
  def __init__(self, head: Optional[ListNode]):
    self.head = head

  def getRandom(self) -> int:
    res = -1
    i = 1
    curr = self.head

    while curr:
      if random.randint(0, i - 1) == 0:
        res = curr.val
      curr = curr.next
      i += 1

    return res


# Link: https://leetcode.com/problems/permutations-ii/description/
class Solution:
  def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    ans = []
    used = [False] * len(nums)

    def dfs(path: List[int]) -> None:
      if len(path) == len(nums):
        ans.append(path.copy())
        return

      for i, num in enumerate(nums):
        if used[i]:
          continue
        if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
          continue
        used[i] = True
        path.append(num)
        dfs(path)
        path.pop()
        used[i] = False

    nums.sort()
    dfs([])
    return ans


# Link: https://leetcode.com/problems/leftmost-column-with-at-least-a-one/description/
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class BinaryMatrix(object):
#   def get(self, row: int, col: int) -> int:
#   def dimensions(self) -> List[int]:

class Solution:
  def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
    m, n = binaryMatrix.dimensions()
    ans = -1
    l = 0
    r = n - 1

    while l <= r:
      mid = (l + r) // 2
      if any(binaryMatrix.get(i, mid) for i in range(m)):
        ans = mid
        r = mid - 1
      else:
        l = mid + 1

    return ans


# Link: https://leetcode.com/problems/leftmost-column-with-at-least-a-one/description/
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class BinaryMatrix(object):
#   def get(self, row: int, col: int) -> int:
#   def dimensions(self) -> List[int]:

class Solution:
  def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
    m, n = binaryMatrix.dimensions()
    ans = -1
    i = 0
    j = n - 1

    while i < m and j >= 0:
      if binaryMatrix.get(i, j):
        ans = j
        j -= 1
      else:
        i += 1

    return ans


# Link: https://leetcode.com/problems/encode-number/description/
class Solution:
  def encode(self, num: int) -> str:
    return bin(num + 1)[3:]


# Link: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/
class Solution:
  def removeDuplicates(self, nums: List[int]) -> int:
    i = 0

    for num in nums:
      if i < 2 or num != nums[i - 2]:
        nums[i] = num
        i += 1

    return i


# Link: https://leetcode.com/problems/add-two-numbers/description/
class Solution:
  def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    carry = 0

    while carry or l1 or l2:
      if l1:
        carry += l1.val
        l1 = l1.next
      if l2:
        carry += l2.val
        l2 = l2.next
      curr.next = ListNode(carry % 10)
      carry //= 10
      curr = curr.next

    return dummy.next


# Link: https://leetcode.com/problems/xor-queries-of-a-subarray/description/
class Solution:
  def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
    ans = []
    xors = [0] * (len(arr) + 1)

    for i, a in enumerate(arr):
      xors[i + 1] = xors[i] ^ a

    for left, right in queries:
      ans.append(xors[left] ^ xors[right + 1])

    return ans


# Link: https://leetcode.com/problems/removing-stars-from-a-string/description/
class Solution:
  def removeStars(self, s: str) -> str:
    ans = []
    for c in s:
      if c == '*':
        ans.pop()
      else:
        ans.append(c)
    return ''.join(ans)


# Link: https://leetcode.com/problems/reconstruct-original-digits-from-english/description/
class Solution:
  def originalDigits(self, s: str) -> str:
    count = [0] * 10

    for c in s:
      if c == 'z':
        count[0] += 1
      if c == 'o':
        count[1] += 1
      if c == 'w':
        count[2] += 1
      if c == 'h':
        count[3] += 1
      if c == 'u':
        count[4] += 1
      if c == 'f':
        count[5] += 1
      if c == 'x':
        count[6] += 1
      if c == 's':
        count[7] += 1
      if c == 'g':
        count[8] += 1
      if c == 'i':
        count[9] += 1

    count[1] -= count[0] + count[2] + count[4]
    count[3] -= count[8]
    count[5] -= count[4]
    count[7] -= count[6]
    count[9] -= count[5] + count[6] + count[8]

    return ''.join(chr(i + ord('0')) for i, c in enumerate(count) for j in range(c))


# Link: https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/
class Solution:
  def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    inToIndex = {num: i for i, num in enumerate(inorder)}

    def build(inStart: int, inEnd: int, postStart: int, postEnd: int) -> Optional[TreeNode]:
      if inStart > inEnd:
        return None

      rootVal = postorder[postEnd]
      rootInIndex = inToIndex[rootVal]
      leftSize = rootInIndex - inStart

      root = TreeNode(rootVal)
      root.left = build(inStart, rootInIndex - 1,  postStart,
                        postStart + leftSize - 1)
      root.right = build(rootInIndex + 1, inEnd,  postStart + leftSize,
                         postEnd - 1)
      return root

    return build(0, len(inorder) - 1, 0, len(postorder) - 1)


# Link: https://leetcode.com/problems/words-within-two-edits-of-dictionary/description/
class Solution:
  def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
    return [query for query in queries
            if any(sum(a != b for a, b in zip(query, word)) < 3
                   for word in dictionary)]


# Link: https://leetcode.com/problems/distinct-numbers-in-each-subarray/description/
class Solution:
  def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
    ans = []
    count = collections.Counter()
    distinct = 0

    for i, num in enumerate(nums):
      count[num] += 1
      if count[num] == 1:
        distinct += 1
      if i >= k:
        count[nums[i - k]] -= 1
        if count[nums[i - k]] == 0:
          distinct -= 1
      if i >= k - 1:
        ans.append(distinct)

    return ans


# Link: https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description/
class Solution:
  def minAddToMakeValid(self, s: str) -> int:
    l = 0
    r = 0

    for c in s:
      if c == '(':
        l += 1
      else:
        if l == 0:
          r += 1
        else:
          l -= 1

    return l + r


# Link: https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/
class Solution:
  def largestValues(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
      return []

    ans = []
    q = collections.deque([root])

    while q:
      maxi = -math.inf
      for _ in range(len(q)):
        root = q.popleft()
        maxi = max(maxi, root.val)
        if root.left:
          q.append(root.left)
        if root.right:
          q.append(root.right)
      ans.append(maxi)

    return ans


# Link: https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/
class Solution:
  def largestValues(self, root: Optional[TreeNode]) -> List[int]:
    ans = []

    def dfs(root: Optional[TreeNode], depth: int) -> None:
      if not root:
        return
      if depth + 1 > len(ans):
        ans.append(root.val)
      else:
        ans[depth] = max(ans[depth], root.val)

      dfs(root.left, depth + 1)
      dfs(root.right, depth + 1)

    dfs(root, 0)
    return ans


# Link: https://leetcode.com/problems/k-concatenation-maximum-sum/description/
class Solution:
  def kConcatenationMaxSum(self, arr: List[int], k: int) -> int:
    kMod = 1_000_000_007
    sz = len(arr) * (1 if k == 1 else 2)
    summ = sum(arr)
    # The concatenated array will be [arr1, arr2, ..., arrk].
    # If sum(arr) > 0 and k > 2, then arr2, ..., arr(k - 1) should be included.
    # Equivalently, maxSubarraySum is from arr1 and arrk.
    if summ > 0 and k > 2:
      return (self.kadane(arr, sz) + summ * (k - 2)) % kMod
    return self.kadane(arr, sz) % kMod

  def kadane(self, A: List[int], sz: int) -> int:
    ans = 0
    summ = 0
    for i in range(sz):
      a = A[i % len(A)]
      summ = max(a, summ + a)
      ans = max(ans, summ)
    return ans


# Link: https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/description/
class Solution:
  def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
    minHeap = list(collections.Counter(arr).values())
    heapq.heapify(minHeap)

    # Greedily remove the k least frequent numbers to have the least number of unique integers.
    while k > 0:
      k -= heapq.heappop(minHeap)

    return len(minHeap) + (1 if k < 0 else 0)


# Link: https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/description/
class Solution:
  def smallestDivisor(self, nums: List[int], threshold: int) -> int:
    l = 1
    r = max(nums)

    while l < r:
      m = (l + r) // 2
      if sum((num - 1) // m + 1 for num in nums) <= threshold:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/apply-operations-to-make-string-empty/description/
class Solution:
  def lastNonEmptyString(self, s: str) -> str:
    ans = []
    count = collections.Counter(s)
    maxFreq = max(count.values())

    for c in reversed(s):
      if count[c] == maxFreq:
        ans.append(c)
        count[c] -= 1

    return ''.join(reversed(ans))


# Link: https://leetcode.com/problems/sum-of-matrix-after-queries/description/
class Solution:
  def matrixSumQueries(self, n: int, queries: List[List[int]]) -> int:
    ans = 0
    # seen[0] := row, seen[1] := col
    seen = [[False] * n for _ in range(2)]
    # notSet[0] = row, notSet[1] := col
    notSet = [n] * 2

    # Late queries dominate.
    for type, index, val in reversed(queries):
      if not seen[type][index]:
        ans += val * notSet[type ^ 1]
        seen[type][index] = True
        notSet[type] -= 1

    return ans


# Link: https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values/description/
class Solution:
  def splitString(self, s: str) -> bool:
    def isValid(s: str, start: int, prev: int, segment: int) -> bool:
      if start == len(s) and segment > 1:
        return True

      curr = 0
      for i in range(start, len(s)):
        curr = curr * 10 + ord(s[i]) - ord('0')
        if curr > 9999999999:
          return False
        if (prev == -1 or curr == prev - 1) and isValid(s, i + 1, curr, segment + 1):
          return True

      return False

    return isValid(s, 0, -1, 0)


# Link: https://leetcode.com/problems/game-of-nim/description/
class Solution:
  def nimGame(self, piles: List[int]) -> bool:
    return functools.reduce(operator.xor, piles) > 0


# Link: https://leetcode.com/problems/time-based-key-value-store/description/
class TimeMap:
  def __init__(self):
    self.values = collections.defaultdict(list)
    self.timestamps = collections.defaultdict(list)

  def set(self, key: str, value: str, timestamp: int) -> None:
    self.values[key].append(value)
    self.timestamps[key].append(timestamp)

  def get(self, key: str, timestamp: int) -> str:
    if key not in self.timestamps:
      return ''
    i = bisect.bisect(self.timestamps[key], timestamp)
    return self.values[key][i - 1] if i > 0 else ''


# Link: https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/description/
class Solution:
  def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if head.val >= 5:
      head = ListNode(0, head)

    curr = head

    while curr:
      curr.val *= 2
      curr.val %= 10
      if curr.next and curr.next.val >= 5:
        curr.val += 1
      curr = curr.next

    return head


# Link: https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/description/
class Solution:
  def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
    def getCarry(node: Optional[ListNode]) -> Optional[ListNode]:
      val = node.val * 2
      if node.next:
        val += getCarry(node.next)
      node.val = val % 10
      return val // 10

    if getCarry(head) == 1:
      return ListNode(1, head)
    return head


# Link: https://leetcode.com/problems/frog-jump-ii/description/
class Solution:
  def maxJump(self, stones: List[int]) -> int:
    # Let's denote the forwarding path as F and the backwarding path as B.
    # 'F1 B2 B1 F2' is no better than 'F1 B2 F2 B1' since the distance between
    # F1 and F2 increase, resulting a larger `ans`.
    if len(stones) == 2:
      return stones[1] - stones[0]
    return max(stones[i] - stones[i - 2]
               for i in range(2, len(stones)))


# Link: https://leetcode.com/problems/network-delay-time/description/
class Solution:
  def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in times:
      graph[u - 1].append((v - 1, w))

    return self._dijkstra(graph, k - 1)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int) -> int:
    dist = [math.inf] * len(graph)

    dist[src] = 0
    minHeap = [(dist[src], src)]  # (d, u)

    while minHeap:
      d, u = heapq.heappop(minHeap)
      for v, w in graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (dist[v], v))

    maxDist = max(dist)
    return maxDist if maxDist != math.inf else -1


# Link: https://leetcode.com/problems/convert-to-base-2/description/
class Solution:
  def baseNeg2(self, n: int) -> str:
    ans = []

    while n != 0:
      ans.append(str(n & 1))
      n = -(n >> 1)

    return ''.join(reversed(ans)) if ans else '0'


# Link: https://leetcode.com/problems/longest-well-performing-interval/description/
class Solution:
  def longestWPI(self, hours: List[int]) -> int:
    ans = 0
    prefix = 0
    dict = {}

    for i in range(len(hours)):
      prefix += 1 if hours[i] > 8 else -1
      if prefix > 0:
        ans = i + 1
      else:
        if prefix not in dict:
          dict[prefix] = i
        if prefix - 1 in dict:
          ans = max(ans, i - dict[prefix - 1])

    return ans


# Link: https://leetcode.com/problems/prime-subtraction-operation/description/
class Solution:
  def primeSubOperation(self, nums: List[int]) -> bool:
    kMax = 1000
    primes = self._sieveEratosthenes(kMax)

    prevNum = 0
    for num in nums:
      # Make nums[i] the smallest as possible and still > nums[i - 1].
      i = bisect.bisect_left(primes, num - prevNum)
      if i > 0:
        num -= primes[i - 1]
      if num <= prevNum:
        return False
      prevNum = num

    return True

  def _sieveEratosthenes(self, n: int) -> List[int]:
    isPrime = [True] * n
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n**0.5) + 1):
      if isPrime[i]:
        for j in range(i * i, n, i):
          isPrime[j] = False
    return [i for i in range(n) if isPrime[i]]


# Link: https://leetcode.com/problems/count-the-number-of-fair-pairs/description/
class Solution:
  def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
    # nums[i] + nums[j] == nums[j] + nums[i], so the condition that i < j
    # degrades to i != j and we can sort the array.
    nums.sort()

    def countLess(summ: int) -> int:
      res = 0
      i = 0
      j = len(nums) - 1
      while i < j:
        while i < j and nums[i] + nums[j] > summ:
          j -= 1
        res += j - i
        i += 1
      return res

    return countLess(upper) - countLess(lower - 1)


# Link: https://leetcode.com/problems/count-number-of-nice-subarrays/description/
class Solution:
  def numberOfSubarrays(self, nums: List[int], k: int) -> int:
    def numberOfSubarraysAtMost(k: int) -> int:
      ans = 0
      l = 0
      r = 0

      while r <= len(nums):
        if k >= 0:
          ans += r - l
          if r == len(nums):
            break
          if nums[r] & 1:
            k -= 1
          r += 1
        else:
          if nums[l] & 1:
            k += 1
          l += 1
      return ans

    return numberOfSubarraysAtMost(k) - numberOfSubarraysAtMost(k - 1)


# Link: https://leetcode.com/problems/find-the-longest-equal-subarray/description/
class Solution:
  def longestEqualSubarray(self, nums: List[int], k: int) -> int:
    ans = 0
    count = collections.Counter()

    # l and r track the maximum window instead of the valid window.
    l = 0
    for r, num in enumerate(nums):
      count[num] += 1
      ans = max(ans, count[num])
      if r - l + 1 - k > ans:
        count[nums[l]] -= 1
        l += 1

    return ans


# Link: https://leetcode.com/problems/find-the-longest-equal-subarray/description/
class Solution:
  def longestEqualSubarray(self, nums: List[int], k: int) -> int:
    ans = 0
    count = collections.Counter()

    l = 0
    for r, num in enumerate(nums):
      count[num] += 1
      ans = max(ans, count[num])
      while r - l + 1 - k > ans:
        count[nums[l]] -= 1
        l += 1

    return ans


# Link: https://leetcode.com/problems/sparse-matrix-multiplication/description/
class Solution:
  def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    m = len(mat1)
    n = len(mat2)
    l = len(mat2[0])
    ans = [[0] * l for _ in range(m)]

    for i in range(m):
      for j in range(l):
        for k in range(n):
          ans[i][j] += mat1[i][k] * mat2[k][j]

    return ans


# Link: https://leetcode.com/problems/sparse-matrix-multiplication/description/
class Solution:
  def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    m = len(mat1)
    n = len(mat2)
    l = len(mat2[0])
    ans = [[0] * l for _ in range(m)]
    nonZeroColIndicesInMat2 = [
        [j for j, a in enumerate(row) if a]
        for row in mat2
    ]

    for i in range(m):
      for j, a in enumerate(mat1[i]):
        if a == 0:
          continue
        # mat1s j-th column matches mat2's j-th row
        for colIndex in nonZeroColIndicesInMat2[j]:
          ans[i][colIndex] += a * mat2[j][colIndex]

    return ans


# Link: https://leetcode.com/problems/longest-uploaded-prefix/description/
class LUPrefix:
  def __init__(self, n: int):
    self.seen = set()
    self.longestPrefix = 0

  def upload(self, video: int) -> None:
    self.seen.add(video)
    while self.longestPrefix + 1 in self.seen:
      self.longestPrefix += 1

  def longest(self) -> int:
    return self.longestPrefix


# Link: https://leetcode.com/problems/flatten-nested-list-iterator/description/
class NestedIterator:
  def __init__(self, nestedList: List[NestedInteger]):
    self.stack: List[NestedInteger] = []
    self.addInteger(nestedList)

  def next(self) -> int:
    return self.stack.pop().getInteger()

  def hasNext(self) -> bool:
    while self.stack and not self.stack[-1].isInteger():
      self.addInteger(self.stack.pop().getList())
    return self.stack

  # addInteger([1, [4, [6]]]) . stack = [[4, [6]], 1]
  # addInteger([4, [6]]) . stack = [[6], 4]
  # addInteger([6]) . stack = [6]
  def addInteger(self, nestedList: List[NestedInteger]) -> None:
    for n in reversed(nestedList):
      self.stack.append(n)


# Link: https://leetcode.com/problems/flatten-nested-list-iterator/description/
class NestedIterator:
  def __init__(self, nestedList: List[NestedInteger]):
    self.q = collections.deque()
    self.addInteger(nestedList)

  def next(self) -> int:
    return self.q.popleft()

  def hasNext(self) -> bool:
    return self.q

  def addInteger(self, nestedList: List[NestedInteger]) -> None:
    for ni in nestedList:
      if ni.isInteger():
        self.q.append(ni.getInteger())
      else:
        self.addInteger(ni.getList())


# Link: https://leetcode.com/problems/search-suggestions-system/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.word: Optional[str] = None


class Solution:
  def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
    ans = []
    root = TrieNode()

    def insert(word: str) -> None:
      node = root
      for c in word:
        node = node.children.setdefault(c, TrieNode())
      node.word = word

    def search(node: Optional[TrieNode]) -> List[str]:
      res: List[str] = []
      dfs(node, res)
      return res

    def dfs(node: Optional[TrieNode], res: List[str]) -> None:
      if len(res) == 3:
        return
      if not node:
        return
      if node.word:
        res.append(node.word)
      for c in string.ascii_lowercase:
        if c in node.children:
          dfs(node.children[c], res)

    for product in products:
      insert(product)

    node = root

    for c in searchWord:
      if not node or c not in node.children:
        node = None
        ans.append([])
        continue
      node = node.children[c]
      ans.append(search(node))

    return ans


# Link: https://leetcode.com/problems/reverse-words-in-a-string-ii/description/
class Solution:
  def reverseWords(self, s: List[str]) -> None:
    def reverse(l: int, r: int) -> None:
      while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1

    def reverseWords(n: int) -> None:
      i = 0
      j = 0

      while i < n:
        while i < j or (i < n and s[i] == ' '):  # Skip the spaces.
          i += 1
        while j < i or (j < n and s[j] != ' '):  # Skip the spaces.
          j += 1
        reverse(i, j - 1)  # Reverse the word.

    reverse(0, len(s) - 1)  # Reverse the whole string.
    reverseWords(len(s))  # Reverse each word.


# Link: https://leetcode.com/problems/decode-xored-permutation/description/
class Solution:
  def decode(self, encoded: List[int]) -> List[int]:
    # Our goal is to find the value of a1, which will allow us to decode a2, a3,
    # ..., an. This can be achieved by performing XOR operation between each
    # element in `encoded` and a1.
    #
    # e.g. n = 3, perm = [a1, a2, a3] is a permutation of [1, 2, 3].
    #               encoded = [a1^a2, a2^a3]
    #    accumulatedEncoded = [a1^a2, a1^a3]
    #    a1 = (a1^a2)^(a1^a3)^(a1^a2^a3)
    #    a2 = a1^(a1^a2)
    #    a3 = a2^(a2^a3)
    n = len(encoded) + 1
    nXors = functools.reduce(operator.xor, [i for i in range(1, n + 1)])

    # Instead of constructing the array, we can track of the running XOR value
    # of `accumulatedEncoded`.
    xors = 0  # xors(accumulatedEncoded)

    for encode in encoded:
      runningXors ^= encode
      xors ^= runningXors

    ans = [xors ^ nXors]

    for encode in encoded:
      ans.append(ans[-1] ^ encode)

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-buy-apples/description/
class Solution:
  def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
    graph = [[] for _ in range(n)]

    for u, v, w in roads:
      graph[u - 1].append((v - 1, w))
      graph[v - 1].append((u - 1, w))

    def dijkstra(i: int) -> int:
      forwardCost = [math.inf] * n
      totalCost = [math.inf] * n
      forwardCost[i] = 0
      q = collections.deque([i])

      while q:
        u = q.popleft()
        for v, w in graph[u]:
          nextCost = forwardCost[u] + w
          if nextCost >= forwardCost[v]:
            continue
          forwardCost[v] = nextCost
          # Take apple at city v and return back to city i.
          totalCost[v] = (k + 1) * nextCost + appleCost[v]
          q.append(v)

      return min(appleCost[i], min(totalCost))

    return [dijkstra(i) for i in range(n)]


# Link: https://leetcode.com/problems/max-consecutive-ones-iii/description/
class Solution:
  def longestOnes(self, nums: List[int], k: int) -> int:
    ans = 0

    l = 0
    for r, num in enumerate(nums):
      if num == 0:
        k -= 1
      while k < 0:
        if nums[l] == 0:
          k += 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/smallest-value-of-the-rearranged-number/description/
class Solution:
  def smallestNumber(self, num: int) -> int:
    s = sorted(str(abs(num)), reverse=num < 0)
    firstNonZeroIndex = next((i for i, c in enumerate(s) if c != '0'), 0)
    s[0], s[firstNonZeroIndex] = s[firstNonZeroIndex], s[0]
    return int(''.join(s)) * (-1 if num < 0 else 1)


# Link: https://leetcode.com/problems/reverse-words-in-a-string/description/
class Solution:
  def reverseWords(self, s: str) -> str:
    return ' '.join(reversed(s.split()))


# Link: https://leetcode.com/problems/maximum-number-of-weeks-for-which-you-can-work/description/
class Solution:
  def numberOfWeeks(self, milestones: List[int]) -> int:
    # The best strategy is to pick 'max, nonMax, max, nonMax, ...'.
    summ = sum(milestones)
    nonMax = summ - max(milestones)
    return min(summ, 2 * nonMax + 1)


# Link: https://leetcode.com/problems/possible-bipartition/description/
from enum import Enum


class Color(Enum):
  kWhite = 0
  kRed = 1
  kGreen = 2


class Solution:
  def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
    graph = [[] for _ in range(n + 1)]
    colors = [Color.kWhite] * (n + 1)

    for u, v in dislikes:
      graph[u].append(v)
      graph[v].append(u)

    # Reduce to 785. Is Graph Bipartite?
    def isValidColor(u: int, color: Color) -> bool:
      # Always paint red for a white node.
      if colors[u] != Color.kWhite:
        return colors[u] == color

      colors[u] = color  # Always paint the node with `color`.

      # All the children should have valid colors.
      childrenColor = Color.kRed if colors[u] == Color.kGreen else Color.kGreen
      return all(isValidColor(v, childrenColor) for v in graph[u])

    return all(colors[i] != Color.kWhite or isValidColor(i, Color.kRed)
               for i in range(1, n + 1))


# Link: https://leetcode.com/problems/partition-equal-subset-sum/description/
class Solution:
  def canPartition(self, nums: List[int]) -> bool:
    summ = sum(nums)
    if summ & 1:
      return False
    return self.knapsack_(nums, summ // 2)

  def knapsack_(self, nums: List[int], subsetSum: int) -> bool:
    n = len(nums)
    # dp[i][j] := True if j can be formed by nums[0..i)
    dp = [[False] * (subsetSum + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
      num = nums[i - 1]
      for j in range(subsetSum + 1):
        if j < num:
          dp[i][j] = dp[i - 1][j]
        else:
          dp[i][j] = dp[i - 1][j] or dp[i - 1][j - num]

    return dp[n][subsetSum]


# Link: https://leetcode.com/problems/partition-equal-subset-sum/description/
class Solution:
  def canPartition(self, nums: List[int]) -> bool:
    summ = sum(nums)
    if summ & 1:
      return False
    return self.knapsack_(nums, summ // 2)

  def knapsack_(self, nums: List[int], subsetSum: int) -> bool:
    # dp[i] := True if i can be formed by nums so far
    dp = [False] * (subsetSum + 1)
    dp[0] = True

    for num in nums:
      for i in range(subsetSum, num - 1, -1):
        dp[i] = dp[i] or dp[i - num]

    return dp[subsetSum]


# Link: https://leetcode.com/problems/sequential-digits/description/
class Solution:
  def sequentialDigits(self, low: int, high: int) -> List[int]:
    ans = []
    q = collections.deque([num for num in range(1, 10)])

    while q:
      num = q.popleft()
      if num > high:
        return ans
      if low <= num and num <= high:
        ans.append(num)
      lastDigit = num % 10
      if lastDigit < 9:
        q.append(num * 10 + lastDigit + 1)

    return ans


# Link: https://leetcode.com/problems/find-the-kth-largest-integer-in-the-array/description/
class Solution:
  # Similar to 215. Kth Largest Element in an Array
  def kthLargestNumber(self, nums: List[str], k: int) -> str:
    minHeap = []

    for num in nums:
      heapq.heappush(minHeap, int(num))
      if len(minHeap) > k:
        heapq.heappop(minHeap)

    return str(minHeap[0])


# Link: https://leetcode.com/problems/satisfiability-of-equality-equations/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))

  def union(self, u: int, v: int) -> None:
    self.id[self.find(u)] = self.find(v)

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def equationsPossible(self, equations: List[str]) -> bool:
    uf = UnionFind(26)

    for x, op, _, y in equations:
      if op == '=':
        uf.union(ord(x) - ord('a'), ord(y) - ord('a'))

    return all(uf.find(ord(x) - ord('a')) != uf.find(ord(y) - ord('a'))
               for x, op, _, y in equations
               if op == '!')


# Link: https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/description/
class Solution:
  def minimumSize(self, nums: List[int], maxOperations: int) -> int:
    # Returns the number of operations required to make m penalty.
    def numOperations(m: int) -> int:
      return sum((num - 1) // m for num in nums) <= maxOperations
    return bisect.bisect_left(range(1, max(nums)), True,
                              key=lambda m: numOperations(m)) + 1


# Link: https://leetcode.com/problems/maximum-candies-allocated-to-k-children/description/
class Solution:
  def maximumCandies(self, candies: List[int], k: int) -> int:
    l = 1
    r = sum(candies) // k

    def numChildren(m: int) -> bool:
      return sum(c // m for c in candies)

    while l < r:
      m = (l + r) // 2
      if numChildren(m) < k:
        r = m
      else:
        l = m + 1

    return l if numChildren(l) >= k else l - 1


# Link: https://leetcode.com/problems/maximum-tastiness-of-candy-basket/description/
class Solution:
  def maximumTastiness(self, price: List[int], k: int) -> int:
    price.sort()

    # Returns true if we can't pick k distinct candies for m tastiness.
    def cantPick(m: int) -> bool:
      baskets = 0
      prevPrice = -m
      for p in price:
        if p >= prevPrice + m:
          prevPrice = p
          baskets += 1
      return baskets < k

    l = bisect.bisect_left(range(max(price) - min(price) + 1), True,
                           key=lambda m: cantPick(m))
    return l - 1


# Link: https://leetcode.com/problems/replace-elements-in-an-array/description/
class Solution:
  def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
    numToIndex = {num: i for i, num in enumerate(nums)}

    for original, replaced in operations:
      index = numToIndex[original]
      nums[index] = replaced
      del numToIndex[original]
      numToIndex[replaced] = index

    return nums


# Link: https://leetcode.com/problems/top-k-frequent-elements/description/
class Solution:
  def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    ans = []
    bucket = [[] for _ in range(len(nums) + 1)]

    for num, freq in collections.Counter(nums).items():
      bucket[freq].append(num)

    for b in reversed(bucket):
      ans += b
      if len(ans) == k:
        return ans


# Link: https://leetcode.com/problems/longest-absolute-file-path/description/
class Solution:
  def lengthLongestPath(self, input: str) -> int:
    ans = 0
    stack = [(-1, 0)]  # placeholder

    for token in input.split('\n'):
      depth = token.count('\t')
      token = token.replace('\t', '')
      while depth <= stack[-1][0]:
        stack.pop()
      if '.' in token:  # `token` is file.
        ans = max(ans, stack[-1][1] + len(token))
      else:  # directory + '/'
        stack.append((depth, stack[-1][1] + len(token) + 1))

    return ans


# Link: https://leetcode.com/problems/longest-consecutive-sequence/description/
class Solution:
  def longestConsecutive(self, nums: List[int]) -> int:
    ans = 0
    seen = set(nums)

    for num in nums:
      # `num` is the start of a sequence.
      if num - 1 in seen:
        continue
      length = 0
      while num in seen:
        num += 1
        length += 1
      ans = max(ans, length)

    return ans


# Link: https://leetcode.com/problems/prison-cells-after-n-days/description/
class Solution:
  def prisonAfterNDays(self, cells: List[int], n: int) -> List[int]:
    nextDayCells = [0] * len(cells)
    day = 0

    while n > 0:
      n -= 1
      for i in range(1, len(cells) - 1):
        nextDayCells[i] = 1 if cells[i - 1] == cells[i + 1] else 0
      if day == 0:
        firstDayCells = nextDayCells.copy()
      elif nextDayCells == firstDayCells:
        n %= day
      cells = nextDayCells.copy()
      day += 1

    return cells


# Link: https://leetcode.com/problems/string-to-integer-atoi/description/
class Solution:
  def myAtoi(self, s: str) -> int:
    s = s.strip()
    if not s:
      return 0

    sign = -1 if s[0] == '-' else 1
    if s[0] in {'-', '+'}:
      s = s[1:]

    num = 0

    for c in s:
      if not c.isdigit():
        break
      num = num * 10 + ord(c) - ord('0')
      if sign * num <= -2**31:
        return -2**31
      if sign * num >= 2**31 - 1:
        return 2**31 - 1

    return sign * num


# Link: https://leetcode.com/problems/design-hit-counter/description/
class HitCounter:
  def __init__(self):
    self.timestamps = [0] * 300
    self.hits = [0] * 300

  def hit(self, timestamp: int) -> None:
    i = timestamp % 300
    if self.timestamps[i] == timestamp:
      self.hits[i] += 1
    else:
      self.timestamps[i] = timestamp
      self.hits[i] = 1  # Reset the hit count to 1.

  def getHits(self, timestamp: int) -> int:
    return sum(h for t, h in zip(self.timestamps, self.hits) if timestamp - t < 300)


# Link: https://leetcode.com/problems/non-decreasing-array/description/
class Solution:
  def checkPossibility(self, nums: List[int]) -> bool:
    j = None

    for i in range(len(nums) - 1):
      if nums[i] > nums[i + 1]:
        if j is not None:
          return False
        j = i

    return j is None or j == 0 or j == len(nums) - 2 or \
        nums[j - 1] <= nums[j + 1] or nums[j] <= nums[j + 2]


# Link: https://leetcode.com/problems/shortest-word-distance-ii/description/
class WordDistance:
  def __init__(self, wordsDict: List[str]):
    self.wordToIndices = collections.defaultdict(list)
    for i, word in enumerate(wordsDict):
      self.wordToIndices[word].append(i)

  def shortest(self, word1: str, word2: str) -> int:
    indices1 = self.wordToIndices[word1]
    indices2 = self.wordToIndices[word2]
    ans = math.inf

    i = 0
    j = 0
    while i < len(indices1) and j < len(indices2):
      ans = min(ans, abs(indices1[i] - indices2[j]))
      if indices1[i] < indices2[j]:
        i += 1
      else:
        j += 1

    return ans


# Link: https://leetcode.com/problems/new-21-game/description/
class Solution:
  def new21Game(self, n: int, k: int, maxPts: int) -> float:
    # When the game ends, the point is in [k..k - 1 maxPts].
    #   P = 1, if n >= k - 1 + maxPts
    #   P = 0, if n < k (note that the constraints already have k <= n)
    if k == 0 or n >= k - 1 + maxPts:
      return 1.0

    ans = 0.0
    dp = [1.0] + [0] * n  # dp[i] := the probability to have i points
    windowSum = dp[0]  # P(i - 1) + P(i - 2) + ... + P(i - maxPts)

    for i in range(1, n + 1):
      # The probability to get i points is
      # P(i) = [P(i - 1) + P(i - 2) + ... + P(i - maxPts)] / maxPts
      dp[i] = windowSum / maxPts
      if i < k:
        windowSum += dp[i]
      else:  # The game ends.
        ans += dp[i]
      if i - maxPts >= 0:
        windowSum -= dp[i - maxPts]

    return ans


# Link: https://leetcode.com/problems/short-encoding-of-words/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.depth = 0


class Solution:
  def minimumLengthEncoding(self, words: List[str]) -> int:
    root = TrieNode()
    leaves = []

    def insert(word: str) -> TrieNode:
      node = root
      for c in reversed(word):
        node = node.children.setdefault(c, TrieNode())
      node.depth = len(word)
      return node

    for word in set(words):
      leaves.append(insert(word))

    return sum(leaf.depth + 1 for leaf in leaves
               if not len(leaf.children))


# Link: https://leetcode.com/problems/maximum-number-of-integers-to-choose-from-a-range-i/description/
class Solution:
  def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
    ans = 0
    summ = 0
    bannedSet = set(banned)

    for i in range(1, n + 1):
      if i not in bannedSet and summ + i <= maxSum:
        ans += 1
        summ += i

    return ans


# Link: https://leetcode.com/problems/steps-to-make-array-non-decreasing/description/
class Solution:
  def totalSteps(self, nums: List[int]) -> int:
    # dp[i] := the number of steps to remove nums[i]
    dp = [0] * len(nums)
    stack = []

    for i, num in enumerate(nums):
      step = 1
      while stack and nums[stack[-1]] <= num:
        step = max(step, dp[stack.pop()] + 1)
      if stack:
        dp[i] = step
      stack.append(i)

    return max(dp)


# Link: https://leetcode.com/problems/the-maze/description/
class Solution:
  def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(maze)
    n = len(maze[0])

    seen = set()

    def isValid(x: int, y: int) -> bool:
      return 0 <= x < m and 0 <= y < n and maze[x][y] == 0

    def dfs(i: int, j: int) -> bool:
      if [i, j] == destination:
        return True
      if (i, j) in seen:
        return False

      seen.add((i, j))

      for dx, dy in dirs:
        x = i
        y = j
        while isValid(x + dx, y + dy):
          x += dx
          y += dy
        if dfs(x, y):
          return True

      return False

    return dfs(start[0], start[1])


# Link: https://leetcode.com/problems/the-maze/description/
class Solution:
  def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(maze)
    n = len(maze[0])
    q = collections.deque([(start[0], start[1])])
    seen = {(start[0], start[1])}

    def isValid(x: int, y: int) -> bool:
      return 0 <= x < m and 0 <= y < n and maze[x][y] == 0

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i
        y = j
        while isValid(x + dx, y + dy):
          x += dx
          y += dy
        if [x, y] == destination:
          return True
        if (x, y) in seen:
          continue
        q.append((x, y))
        seen.add((x, y))

    return False


# Link: https://leetcode.com/problems/longest-repeating-character-replacement/description/
class Solution:
  def characterReplacement(self, s: str, k: int) -> int:
    ans = 0
    maxCount = 0
    count = collections.Counter()

    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      maxCount = max(maxCount, count[c])
      while maxCount + k < r - l + 1:
        count[s[l]] -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/longest-repeating-character-replacement/description/
class Solution:
  def characterReplacement(self, s: str, k: int) -> int:
    maxCount = 0
    count = collections.Counter()

    # l and r track the maximum window instead of the valid window.
    l = 0
    for r, c in enumerate(s):
      count[c] += 1
      maxCount = max(maxCount, count[c])
      while maxCount + k < r - l + 1:
        count[s[l]] -= 1
        l += 1

    return r - l + 1


# Link: https://leetcode.com/problems/zigzag-conversion/description/
class Solution:
  def convert(self, s: str, numRows: int) -> str:
    rows = [''] * numRows
    k = 0
    direction = (numRows == 1) - 1

    for c in s:
      rows[k] += c
      if k == 0 or k == numRows - 1:
        direction *= -1
      k += direction

    return ''.join(rows)


# Link: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/description/
class Solution:
  def maxProduct(self, s: str) -> int:
    ans = 0

    def isPalindrome(s: str) -> bool:
      i = 0
      j = len(s) - 1
      while i < j:
        if s[i] != s[j]:
          return False
        i += 1
        j -= 1
      return True

    @functools.lru_cache(None)
    def dfs(i: int, s1: str, s2: str) -> None:
      nonlocal ans
      if i == len(s):
        if isPalindrome(s1) and isPalindrome(s2):
          ans = max(ans, len(s1) * len(s2))
        return

      dfs(i + 1, s1 + s[i], s2)
      dfs(i + 1, s1, s2 + s[i])
      dfs(i + 1, s1, s2)

    dfs(0, '', '')
    return ans


# Link: https://leetcode.com/problems/replace-the-substring-for-balanced-string/description/
class Solution:
  def balancedString(self, s: str) -> int:
    ans = len(s)
    count = collections.Counter(s)
    j = 0

    for i, c in enumerate(s):
      count[c] -= 1
      while j < len(s) and all(count[c] <= len(s) // 4 for c in 'QWER'):
        ans = min(ans, i - j + 1)
        count[s[j]] += 1
        j += 1

    return ans


# Link: https://leetcode.com/problems/count-strictly-increasing-subarrays/description/
class Solution:
  def countSubarrays(self, nums: List[int]) -> int:
    ans = 0

    j = -1
    for i, num in enumerate(nums):
      if i > 0 and num <= nums[i - 1]:
        j = i - 1
      ans += i - j

    return ans


# Link: https://leetcode.com/problems/number-of-same-end-substrings/description/
class Solution:
  def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
    count = collections.Counter()
    # counts[i] := the count of s[0..i)
    counts = [count.copy()]

    for c in s:
      count[c] += 1
      counts.append(count.copy())

    ans = []

    for l, r in queries:
      sameEndCount = 0
      for c in string.ascii_lowercase:
        #   the count of s[0..r] - the count of s[0..l - 1]
        # = the count of s[l..r]
        freq = counts[r + 1][c] - counts[l][c]
        #   C(freq, 2) + freq
        # = freq * (freq - 1) / 2 + freq
        # = freq * (freq + 1) / 2
        sameEndCount += freq * (freq + 1) // 2
      ans.append(sameEndCount)

    return ans


# Link: https://leetcode.com/problems/remove-nodes-from-linked-list/description/
class Solution:
  def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
      return None
    head.next = self.removeNodes(head.next)
    return head.next if head.next and head.val < head.next.val else head


# Link: https://leetcode.com/problems/find-the-longest-semi-repetitive-substring/description/
class Solution:
  def longestSemiRepetitiveSubstring(self, s: str) -> int:
    ans = 1
    prevStart = 0
    start = 0

    for i in range(1, len(s)):
      if s[i] == s[i - 1]:
        if prevStart > 0:
          start = prevStart
        prevStart = i
      ans = max(ans, i - start + 1)

    return ans


# Link: https://leetcode.com/problems/simplify-path/description/
class Solution:
  def simplifyPath(self, path: str) -> str:
    stack = []

    for str in path.split('/'):
      if str in ('', '.'):
        continue
      if str == '..':
        if stack:
          stack.pop()
      else:
        stack.append(str)

    return '/' + '/'.join(stack)


# Link: https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/description/
class Solution:
  def longestBeautifulSubstring(self, word: str) -> int:
    ans = 0
    count = 1

    l = 0
    for r in range(1, len(word)):
      curr = word[r]
      prev = word[r - 1]
      if curr >= prev:
        if curr > prev:
          count += 1
        if count == 5:
          ans = max(ans, r - l + 1)
      else:
        count = 1
        l = r

    return ans


# Link: https://leetcode.com/problems/find-all-lonely-numbers-in-the-array/description/
class Solution:
  def findLonely(self, nums: List[int]) -> List[int]:
    count = collections.Counter(nums)
    return [num for num, freq in count.items()
            if freq == 1 and
            count[num - 1] == 0 and
            count[num + 1] == 0]


# Link: https://leetcode.com/problems/longest-palindromic-substring/description/
class Solution:
  def longestPalindrome(self, s: str) -> str:
    # '@' and '$' signs serve as sentinels appended to each end to avoid bounds
    # checking.
    t = '#'.join('@' + s + '$')
    n = len(t)
    # t[i - maxExtends[i]..i) ==
    # t[i + 1..i + maxExtends[i]]
    maxExtends = [0] * n
    center = 0

    for i in range(1, n - 1):
      rightBoundary = center + maxExtends[center]
      mirrorIndex = center - (i - center)
      maxExtends[i] = rightBoundary > i and \
          min(rightBoundary - i, maxExtends[mirrorIndex])

      # Attempt to expand the palindrome centered at i.
      while t[i + 1 + maxExtends[i]] == t[i - 1 - maxExtends[i]]:
        maxExtends[i] += 1

      # If a palindrome centered at i expand past `rightBoundary`, adjust
      # center based on expanded palindrome.
      if i + maxExtends[i] > rightBoundary:
        center = i

    # Find `maxExtend` and `bestCenter`.
    maxExtend, bestCenter = max((extend, i)
                                for i, extend in enumerate(maxExtends))
    return s[(bestCenter - maxExtend) // 2:(bestCenter + maxExtend) // 2]


# Link: https://leetcode.com/problems/longest-palindromic-substring/description/
class Solution:
  def longestPalindrome(self, s: str) -> str:
    if not s:
      return ''

    # (start, end) indices of the longest palindrome in s
    indices = [0, 0]

    def extend(s: str, i: int, j: int) -> Tuple[int, int]:
      """
      Returns the (start, end) indices of the longest palindrome extended from
      the substring s[i..j].
      """
      while i >= 0 and j < len(s):
        if s[i] != s[j]:
          break
        i -= 1
        j += 1
      return i + 1, j - 1

    for i in range(len(s)):
      l1, r1 = extend(s, i, i)
      if r1 - l1 > indices[1] - indices[0]:
        indices = l1, r1
      if i + 1 < len(s) and s[i] == s[i + 1]:
        l2, r2 = extend(s, i, i + 1)
        if r2 - l2 > indices[1] - indices[0]:
          indices = l2, r2

    return s[indices[0]:indices[1] + 1]


# Link: https://leetcode.com/problems/next-greater-node-in-linked-list/description/
class Solution:
  def nextLargerNodes(self, head: ListNode) -> List[int]:
    ans = []
    stack = []

    while head:
      while stack and head.val > ans[stack[-1]]:
        index = stack.pop()
        ans[index] = head.val
      stack.append(len(ans))
      ans.append(head.val)
      head = head.next

    for i in stack:
      ans[i] = 0

    return ans


# Link: https://leetcode.com/problems/max-number-of-k-sum-pairs/description/
class Solution:
  def maxOperations(self, nums: List[int], k: int) -> int:
    count = collections.Counter(nums)
    return sum(min(count[num], count[k - num])
               for num in count) // 2


# Link: https://leetcode.com/problems/factorial-trailing-zeroes/description/
class Solution:
  def trailingZeroes(self, n: int) -> int:
    return 0 if n == 0 else n // 5 + self.trailingZeroes(n // 5)


# Link: https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/
class Solution:
  def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    p = None  # Points the k-th node from the beginning.
    q = None  # Points the k-th node from the end.

    curr = head
    while curr:
      if q:
        q = q.next
      k -= 1
      if k == 0:
        p = curr
        q = head
      curr = curr.next

    p.val, q.val = q.val, p.val
    return head


# Link: https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/description/
class Solution:
  def findSmallestInteger(self, nums: List[int], value: int) -> int:
    count = collections.Counter([num % value for num in nums])

    for i in range(len(nums)):
      if count[i % value] == 0:
        return i
      count[i % value] -= 1

    return len(nums)


# Link: https://leetcode.com/problems/buildings-with-an-ocean-view/description/
class Solution:
  def findBuildings(self, heights: List[int]) -> List[int]:
    stack = []

    for i, height in enumerate(heights):
      while stack and heights[stack[-1]] <= height:
        stack.pop()
      stack.append(i)

    return stack


# Link: https://leetcode.com/problems/merge-in-between-linked-lists/description/
class Solution:
  def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    nodeBeforeA = list1
    for i in range(a - 1):
      nodeBeforeA = nodeBeforeA.next

    nodeB = nodeBeforeA.next
    for i in range(b - a):
      nodeB = nodeB.next

    nodeBeforeA.next = list2
    lastNodeInList2 = list2

    while lastNodeInList2.next:
      lastNodeInList2 = lastNodeInList2.next

    lastNodeInList2.next = nodeB.next
    nodeB.next = None
    return list1


# Link: https://leetcode.com/problems/largest-merge-of-two-strings/description/
class Solution:
  def largestMerge(self, word1: str, word2: str) -> str:
    if not word1:
      return word2
    if not word2:
      return word1
    if word1 > word2:
      return word1[0] + self.largestMerge(word1[1:], word2)
    return word2[0] + self.largestMerge(word1, word2[1:])


# Link: https://leetcode.com/problems/count-collisions-of-monkeys-on-a-polygon/description/
class Solution:
  def monkeyMove(self, n: int) -> int:
    kMod = 1_000_000_007
    res = pow(2, n, kMod) - 2
    return res + kMod if res < 0 else res


# Link: https://leetcode.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/description/
class Solution:
  def hasAllCodes(self, s: str, k: int) -> bool:
    n = 1 << k
    if len(s) < n:
      return False

    # used[i] := True if i is a substring of `s`
    used = [0] * n

    windowStr = 0 if k == 1 else int(s[0:k - 1], 2)
    for i in range(k - 1, len(s)):
      # Include the s[i].
      windowStr = (windowStr << 1) + int(s[i])
      # Discard the s[i - k].
      windowStr &= n - 1
      used[windowStr] = True

    return all(u for u in used)


# Link: https://leetcode.com/problems/range-sum-query-mutable/description/
class FenwickTree:
  def __init__(self, n: int):
    self.sums = [0] * (n + 1)

  def update(self, i: int, delta: int) -> None:
    while i < len(self.sums):
      self.sums[i] += delta
      i += FenwickTree.lowbit(i)

  def get(self, i: int) -> int:
    summ = 0
    while i > 0:
      summ += self.sums[i]
      i -= FenwickTree.lowbit(i)
    return summ

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i


class NumArray:
  def __init__(self, nums: List[int]):
    self.nums = nums
    self.tree = FenwickTree(len(nums))
    for i, num in enumerate(nums):
      self.tree.update(i + 1, num)

  def update(self, index: int, val: int) -> None:
    self.tree.update(index + 1, val - self.nums[index])
    self.nums[index] = val

  def sumRange(self, left: int, right: int) -> int:
    return self.tree.get(right + 1) - self.tree.get(left)


# Link: https://leetcode.com/problems/remove-covered-intervals/description/
class Solution:
  def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
    ans = 0
    prevEnd = 0

    # If the two intervals have the same `start`, put the one with a larger
    # `end` first.
    for _, end in sorted(intervals, key=lambda x: (x[0], -x[1])):
      # Current interval is not covered by the previous one.
      if prevEnd < end:
        ans += 1
        prevEnd = end

    return ans


# Link: https://leetcode.com/problems/flip-game-ii/description/
class Solution:
  @functools.lru_cache(None)
  def canWin(self, currentState: str) -> bool:
    # If any of currentState[i:i + 2] == "++" and your friend can't win after
    # changing currentState[i:i + 2] to "--" (or "-"), then you can win.
    return any(True
               for i, (a, b) in enumerate(zip(currentState, currentState[1:]))
               if a == '+' and b == '+' and
               not self.canWin(currentState[:i] + '-' + currentState[i + 2:]))


# Link: https://leetcode.com/problems/line-reflection/description/
class Solution:
  def isReflected(self, points: List[List[int]]) -> bool:
    minX = math.inf
    maxX = -math.inf
    seen = set()

    for x, y in points:
      minX = min(minX, x)
      maxX = max(maxX, x)
      seen.add((x, y))

    summ = minX + maxX
    # (leftX + rightX) / 2 = (minX + maxX) / 2
    #  leftX = minX + maxX - rightX
    # rightX = minX + maxX - leftX

    return all((summ - x, y) in seen for x, y in points)


# Link: https://leetcode.com/problems/android-unlock-patterns/description/
class Solution:
  def numberOfPatterns(self, m: int, n: int) -> int:
    seen = set()
    accross = [[0] * 10 for _ in range(10)]

    accross[1][3] = accross[3][1] = 2
    accross[1][7] = accross[7][1] = 4
    accross[3][9] = accross[9][3] = 6
    accross[7][9] = accross[9][7] = 8
    accross[1][9] = accross[9][1] = accross[2][8] = accross[8][2] = \
        accross[3][7] = accross[7][3] = accross[4][6] = accross[6][4] = 5

    def dfs(u: int, depth: int) -> int:
      if depth > n:
        return 0

      seen.add(u)
      ans = 1 if depth >= m else 0

      for v in range(1, 10):
        if v == u or v in seen:
          continue
        accrossed = accross[u][v]
        if not accrossed or accrossed in seen:
          ans += dfs(v, depth + 1)

      seen.remove(u)
      return ans

    # 1, 3, 7, 9 are symmetric
    # 2, 4, 6, 8 are symmetric
    return dfs(1, 1) * 4 + dfs(2, 1) * 4 + dfs(5, 1)


# Link: https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs-ii/description/
class Solution:
  def minimumTime(self, jobs: List[int], workers: List[int]) -> int:
    ans = 0

    jobs.sort()
    workers.sort()

    for job, worker in zip(jobs, workers):
      ans = max(ans, (job - 1) // worker + 1)

    return ans


# Link: https://leetcode.com/problems/longest-increasing-subsequence/description/
class Solution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    # tails[i] := the minimum tails of all the increasing subsequences having
    # length i + 1
    # It's easy to see that `tails` must be an increasing array.
    tails = []

    for num in nums:
      if not tails or num > tails[-1]:
        tails.append(num)
      else:
        tails[bisect.bisect_left(tails, num)] = num

    return len(tails)


# Link: https://leetcode.com/problems/longest-increasing-subsequence/description/
class Solution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    if not nums:
      return 0

    # dp[i] the length of LIS ending in nums[i]
    dp = [1] * len(nums)

    for i in range(1, len(nums)):
      for j in range(i):
        if nums[j] < nums[i]:
          dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


# Link: https://leetcode.com/problems/peeking-iterator/description/
class PeekingIterator:
  def __init__(self, iterator: Iterator):
    self.iterator = iterator
    self.buffer = self.iterator.next() if self.iterator.hasNext() else None

  def peek(self) -> int:
    """
    Returns the next element in the iteration without advancing the iterator.
    """
    return self.buffer

  def next(self) -> int:
    next = self.buffer
    self.buffer = self.iterator.next() if self.iterator.hasNext() else None
    return next

  def hasNext(self) -> bool:
    return self.buffer is not None


# Link: https://leetcode.com/problems/the-earliest-moment-when-everyone-become-friends/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.rank[i] < self.rank[j]:
      self.id[i] = j
    elif self.rank[i] > self.rank[j]:
      self.id[j] = i
    else:
      self.id[i] = j
      self.rank[j] += 1
    self.count -= 1

  def getCount(self) -> int:
    return self.count

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def earliestAcq(self, logs: List[List[int]], n: int) -> int:
    uf = UnionFind(n)

    logs.sort(key=lambda x: x[0])

    # Sort `logs` by timestamp.
    for timestamp, x, y in logs:
      uf.unionByRank(x, y)
      if uf.getCount() == 1:
        return timestamp

    return -1


# Link: https://leetcode.com/problems/minimum-operations-to-halve-array-sum/description/
class Solution:
  def halveArray(self, nums: List[int]) -> int:
    halfSum = sum(nums) / 2
    ans = 0
    runningSum = 0.0
    maxHeap = [-num for num in nums]

    heapq.heapify(maxHeap)

    while runningSum < halfSum:
      maxValue = -heapq.heappop(maxHeap) / 2
      runningSum += maxValue
      heapq.heappush(maxHeap, -maxValue)
      ans += 1

    return ans


# Link: https://leetcode.com/problems/closest-nodes-queries-in-a-binary-search-tree/description/
class Solution:
  def closestNodes(self, root: Optional[TreeNode], queries: List[int]) -> List[List[int]]:
    sortedVals = []
    self._inorder(root, sortedVals)

    def getClosestPair(query: int) -> List[int]:
      i = bisect_left(sortedVals, query)
      # query is presented in the tree, so just use [query, query].
      if i != len(sortedVals) and sortedVals[i] == query:
        return [query, query]
      # query isn't presented in the tree, so find the cloest one if possible.
      return [-1 if i == 0 else sortedVals[i - 1],
              -1 if i == len(sortedVals) else sortedVals[i]]

    return [getClosestPair(query) for query in queries]

  def _inorder(self, root: Optional[TreeNode], sortedVals: List[int]) -> None:
    """Walks the BST to collect the sorted numbers."""
    if not root:
      return
    self._inorder(root.left, sortedVals)
    sortedVals.append(root.val)
    self._inorder(root.right, sortedVals)


# Link: https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid/description/
class Solution:
  def minOperations(self, grid: List[List[int]], x: int) -> int:
    A = sorted([a for row in grid for a in row])
    if any((a - A[0]) % x for a in A):
      return -1

    ans = 0

    for a in A:
      ans += abs(a - A[len(A) // 2]) // x

    return ans


# Link: https://leetcode.com/problems/guess-the-majority-in-a-hidden-array/description/
# """
# This is the ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
# class ArrayReader(object):
#   # Compares 4 different elements in the array
#   # Returns 4 if the values of the 4 elements are the same (0 or 1).
#   # Returns 2 if three elements have a value equal to 0 and one element has
#   #           value equal to 1 or vice versa.
#   # Returns 0 if two element have a value equal to 0 and two elements have a
#   #           value equal to 1.
#   def query(self, a: int, b: int, c: int, d: int) -> int:
#
#   # Returns the length of the array
#   def length(self) -> int:
#

class Solution:
  def guessMajority(self, reader: 'ArrayReader') -> int:
    n = reader.length()
    query0123 = reader.query(0, 1, 2, 3)
    query1234 = reader.query(1, 2, 3, 4)
    count0 = 1  # the number of numbers that are same as `nums[0]`
    countNot0 = 0  # the number of numbers that are different from `nums[0]`
    indexNot0 = -1  # any index i s.t. nums[i] != nums[0]

    # Find which group nums[1..3] belong to.
    for i in range(1, 4):
      abcd = [0] + [num for num in [1, 2, 3] if num != i] + [4]
      if reader.query(*abcd) == query1234:  # nums[i] == nums[0]
        count0 += 1
      else:
        countNot0 += 1
        indexNot0 = i

    # Find which group nums[4..n) belong to.
    for i in range(4, n):
      if reader.query(1, 2, 3, i) == query0123:  # nums[i] == nums[0]
        count0 += 1
      else:
        countNot0 += 1
        indexNot0 = i

    if count0 == countNot0:
      return -1
    if count0 > countNot0:
      return 0
    return indexNot0


# Link: https://leetcode.com/problems/largest-palindromic-number/description/
class Solution:
  def largestPalindromic(self, num: str) -> str:
    count = collections.Counter(num)
    firstHalf = ''.join(count[i] // 2 * i for i in '9876543210').lstrip('0')
    mid = self._getMid(count)
    return (firstHalf + mid + firstHalf[::-1]) or '0'

  def _getMid(self, count: Dict[str, int]) -> str:
    for c in '9876543210':
      if count[c] & 1:
        return c
    return ''


# Link: https://leetcode.com/problems/course-schedule/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = [[] for _ in range(numCourses)]
    states = [State.kInit] * numCourses

    for v, u in prerequisites:
      graph[u].append(v)

    def hasCycle(u: int) -> bool:
      if states[u] == State.kVisiting:
        return True
      if states[u] == State.kVisited:
        return False

      states[u] = State.kVisiting
      if any(hasCycle(v) for v in graph[u]):
        return True
      states[u] = State.kVisited

      return False

    return not any(hasCycle(i) for i in range(numCourses))


# Link: https://leetcode.com/problems/maximum-price-to-fill-a-bag/description/
class Solution:
  def maxPrice(self, items: List[List[int]], capacity: int) -> float:
    ans = 0

    # Sort items based on price//weight.
    for price, weight in sorted(items, key=lambda x: -x[0] / x[1]):
      # The bag is filled.
      if capacity <= weight:
        return ans + price * capacity / weight
      ans += price
      capacity -= weight

    return -1


# Link: https://leetcode.com/problems/elements-in-array-after-removing-and-replacing-elements/description/
class Solution:
  def elementInNums(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    n = len(nums)

    def f(time: int, index: int) -> int:
      if time < n:  # [0, 1, 2] -> [1, 2] -> [2]
        index += time
        return -1 if index >= n else nums[index]
      else:  # [] -> [0] -> [0, 1]
        return -1 if index >= time - n else nums[index]

    return [f(time % (2 * n), index) for time, index in queries]


# Link: https://leetcode.com/problems/groups-of-special-equivalent-strings/description/
class Solution:
  def numSpecialEquivGroups(self, words: List[str]) -> int:
    return len({''.join(sorted(word[::2])) + ''.join(sorted(word[1::2]))
                for word in words})


# Link: https://leetcode.com/problems/count-univalue-subtrees/description/
class Solution:
  def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
    ans = 0

    def isUnival(root: Optional[TreeNode], val: int) -> bool:
      nonlocal ans
      if not root:
        return True

      if isUnival(root.left, root.val) & isUnival(root.right, root.val):
        ans += 1
        return root.val == val

      return False

    isUnival(root, math.inf)
    return ans


# Link: https://leetcode.com/problems/sum-of-square-numbers/description/
class Solution:
  def judgeSquareSum(self, c: int) -> bool:
    l = 0
    r = int(sqrt(c))

    while l <= r:
      summ = l * l + r * r
      if summ == c:
        return True
      if summ < c:
        l += 1
      else:
        r -= 1

    return False


# Link: https://leetcode.com/problems/bag-of-tokens/description/
class Solution:
  def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
    ans = 0
    score = 0
    q = collections.deque(sorted(tokens))

    while q and (power >= q[0] or score):
      while q and power >= q[0]:
        # Play the smallest face up.
        power -= q.popleft()
        score += 1
      ans = max(ans, score)
      if q and score:
        # Play the largest face down.
        power += q.pop()
        score -= 1

    return ans


# Link: https://leetcode.com/problems/zigzag-iterator/description/
class ZigzagIterator:
  def __init__(self, v1: List[int], v2: List[int]):
    def vals():
      for i in itertools.count():
        for v in v1, v2:
          if i < len(v):
            yield v[i]
    self.vals = vals()
    self.n = len(v1) + len(v2)

  def next(self):
    self.n -= 1
    return next(self.vals)

  def hasNext(self):
    return self.n > 0


# Link: https://leetcode.com/problems/inorder-successor-in-bst/description/
class Solution:
  def inorderSuccessor(self, root: Optional[TreeNode], p: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
      return None
    if root.val <= p.val:
      return self.inorderSuccessor(root.right, p)
    return self.inorderSuccessor(root.left, p) or root


# Link: https://leetcode.com/problems/increasing-triplet-subsequence/description/
class Solution:
  def increasingTriplet(self, nums: List[int]) -> bool:
    first = math.inf
    second = math.inf

    for num in nums:
      if num <= first:
        first = num
      elif num <= second:  # first < num <= second
        second = num
      else:
        return True  # first < second < num (third)

    return False


# Link: https://leetcode.com/problems/construct-string-from-binary-tree/description/
class Solution:
  def tree2str(self, t: Optional[TreeNode]) -> str:
    def dfs(root: Optional[TreeNode]) -> str:
      if not root:
        return ''
      if root.right:
        return str(root.val) + '(' + dfs(root.left) + ')(' + dfs(root.right) + ')'
      if root.left:
        return str(root.val) + '(' + dfs(root.left) + ')'
      return str(root.val)
    return dfs(t)


# Link: https://leetcode.com/problems/sort-colors/description/
class Solution:
  def sortColors(self, nums: List[int]) -> None:
    l = 0  # The next 0 should be placed in l.
    r = len(nums) - 1  # THe next 2 should be placed in r.

    i = 0
    while i <= r:
      if nums[i] == 0:
        nums[i], nums[l] = nums[l], nums[i]
        i += 1
        l += 1
      elif nums[i] == 1:
        i += 1
      else:
        # We may swap a 0 to index i, but we're still not sure whether this 0
        # is placed in the correct index, so we can't move pointer i.
        nums[i], nums[r] = nums[r], nums[i]
        r -= 1


# Link: https://leetcode.com/problems/sort-colors/description/
class Solution:
  def sortColors(self, nums: List[int]) -> None:
    zero = -1
    one = -1
    two = -1

    for num in nums:
      if num == 0:
        two += 1
        one += 1
        zero += 1
        nums[two] = 2
        nums[one] = 1
        nums[zero] = 0
      elif num == 1:
        two += 1
        one += 1
        nums[two] = 2
        nums[one] = 1
      else:
        two += 1
        nums[two] = 2


# Link: https://leetcode.com/problems/reorder-list/description/
class Solution:
  def reorderList(self, head: ListNode) -> None:
    def findMid(head: ListNode):
      prev = None
      slow = head
      fast = head

      while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
      prev.next = None

      return slow

    def reverse(head: ListNode) -> ListNode:
      prev = None
      curr = head

      while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

      return prev

    def merge(l1: ListNode, l2: ListNode) -> None:
      while l2:
        next = l1.next
        l1.next = l2
        l1 = l2
        l2 = next

    if not head or not head.next:
      return

    mid = findMid(head)
    reversed = reverse(mid)
    merge(head, reversed)


# Link: https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/
class Solution:
  def findMinArrowShots(self, points: List[List[int]]) -> int:
    ans = 0
    arrowX = -math.inf

    for point in sorted(points, key=lambda x: x[1]):
      if point[0] > arrowX:
        ans += 1
        arrowX = point[1]

    return ans


# Link: https://leetcode.com/problems/sort-characters-by-frequency/description/
class Solution:
  def frequencySort(self, s: str) -> str:
    ans = []
    buckets = [[] for _ in range(len(s) + 1)]

    for c, freq in collections.Counter(s).items():
      buckets[freq].append(c)

    for freq in reversed(range(len(buckets))):
      for c in buckets[freq]:
        ans.append(c * freq)

    return ''.join(ans)


# Link: https://leetcode.com/problems/check-if-two-expression-trees-are-equivalent/description/
class Solution:
  def checkEquivalence(self, root1: 'Node', root2: 'Node') -> bool:
    count = collections.Counter()

    def dfs(root: 'Node', add: int) -> None:
      if not root:
        return
      if 'a' <= root.val <= 'z':
        count[root.val] += add
      dfs(root.left, add)
      dfs(root.right, add)

    dfs(root1, 1)
    dfs(root2, -1)
    return all(value == 0 for value in count.values())


# Link: https://leetcode.com/problems/search-a-2d-matrix/description/
class Solution:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix:
      return False

    m = len(matrix)
    n = len(matrix[0])
    l = 0
    r = m * n

    while l < r:
      mid = (l + r) // 2
      i = mid // n
      j = mid % n
      if matrix[i][j] == target:
        return True
      if matrix[i][j] < target:
        l = mid + 1
      else:
        r = mid

    return False


# Link: https://leetcode.com/problems/range-frequency-queries/description/
class RangeFreqQuery:
  def __init__(self, arr: List[int]):
    self.valueToIndices = collections.defaultdict(list)
    for i, a in enumerate(arr):
      self.valueToIndices[a].append(i)

  def query(self, left: int, right: int, value: int) -> int:
    indices = self.valueToIndices[value]
    i = bisect_left(indices, left)
    j = bisect_right(indices, right)
    return j - i


# Link: https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments/description/
class Solution:
  def numberOfSets(self, n: int, k: int) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(i: int, k: int, drawing: bool) -> int:
      if k == 0:  # Find a way to draw k segments.
        return 1
      if i == n:  # Reach the end.
        return 0
      if drawing:
        # 1. Keep drawing at i and move to i + 1.
        # 2. Stop at i so decrease k. We can start from i for the next segment.
        return (dp(i + 1, k, True) + dp(i, k - 1, False)) % kMod
      # 1. Skip i and move to i + 1.
      # 2. Start at i and move to i + 1.
      return (dp(i + 1, k, False) + dp(i + 1, k, True)) % kMod

    return dp(0, k, False)


# Link: https://leetcode.com/problems/rotate-list/description/
class Solution:
  def rotateRight(self, head: ListNode, k: int) -> ListNode:
    if not head or not head.next or k == 0:
      return head

    tail = head
    length = 1
    while tail.next:
      tail = tail.next
      length += 1
    tail.next = head  # Circle the list.

    t = length - k % length
    for _ in range(t):
      tail = tail.next
    newHead = tail.next
    tail.next = None

    return newHead


# Link: https://leetcode.com/problems/incremental-memory-leak/description/
class Solution:
  def memLeak(self, memory1: int, memory2: int) -> List[int]:
    i = 1

    while memory1 >= i or memory2 >= i:
      if memory1 >= memory2:
        memory1 -= i
      else:
        memory2 -= i
      i += 1

    return [i, memory1, memory2]


# Link: https://leetcode.com/problems/maximum-xor-after-operations/description/
class Solution:
  def maximumXOR(self, nums: List[int]) -> int:
    # 1. nums[i] & (nums[i] ^ x) enables you to turn 1-bit to 0-bit from
    #    nums[i] since x is arbitrary.
    # 2. The i-th bit of the XOR of all the elements is 1 if the i-th bit is 1
    #    for an odd number of elements.
    # 3. Therefore, the question is equivalent to: if you can convert any digit
    #    from 1 to 0 for any number, what is the maximum for XOR(nums[i]).
    # 4. The maximum we can get is of course to make every digit of the answer
    #    to be 1 if possible
    # 5. Therefore, OR(nums[i]) is an approach.
    return functools.reduce(operator.ior, nums)


# Link: https://leetcode.com/problems/count-complete-subarrays-in-an-array/description/
class Solution:
  def countCompleteSubarrays(self, nums: List[int]) -> int:
    ans = 0
    distinct = len(set(nums))
    count = collections.Counter()

    l = 0
    for num in nums:
      count[num] += 1
      while len(count) == distinct:
        count[nums[l]] -= 1
        if count[nums[l]] == 0:
          del count[nums[l]]
        l += 1
      # Assume nums[r] = num,
      # nums[0..r], nums[1..r], ..., nums[l - 1..r] have k different values.
      ans += l

    return ans


# Link: https://leetcode.com/problems/min-stack/description/
class MinStack:
  def __init__(self):
    self.stack = []

  def push(self, x: int) -> None:
    mini = x if not self.stack else min(self.stack[-1][1], x)
    self.stack.append([x, mini])

  def pop(self) -> None:
    self.stack.pop()

  def top(self) -> int:
    return self.stack[-1][0]

  def getMin(self) -> int:
    return self.stack[-1][1]


# Link: https://leetcode.com/problems/sort-an-array/description/
class Solution:
  def sortArray(self, nums: List[int]) -> List[int]:
    def heapSort(A: List[int]) -> None:
      def maxHeapify(A: List[int], i: int, heapSize: int) -> None:
        l = 2 * i + 1
        r = 2 * i + 2
        largest = i
        if l < heapSize and A[largest] < A[l]:
          largest = l
        if r < heapSize and A[largest] < A[r]:
          largest = r
        if largest != i:
          A[largest], A[i] = A[i], A[largest]
          maxHeapify(A, largest, heapSize)

      def buildMaxHeap(A: List[int]) -> None:
        for i in range(len(A) // 2, -1, -1):
          maxHeapify(A, i, len(A))

      buildMaxHeap(A)
      heapSize = len(A)
      for i in reversed(range(1, len(A))):
        A[i], A[0] = A[0], A[i]
        heapSize -= 1
        maxHeapify(A, 0, heapSize)

    heapSort(nums)
    return nums


# Link: https://leetcode.com/problems/sort-an-array/description/
class Solution:
  def sortArray(self, nums: List[int]) -> List[int]:
    def quickSort(A: List[int], l: int, r: int) -> None:
      if l >= r:
        return

      def partition(A: List[int], l: int, r: int) -> int:
        randIndex = random.randint(0, r - l) + l
        A[randIndex], A[r] = A[r], A[randIndex]
        pivot = A[r]
        nextSwapped = l
        for i in range(l, r):
          if A[i] <= pivot:
            A[nextSwapped], A[i] = A[i], A[nextSwapped]
            nextSwapped += 1
        A[nextSwapped], A[r] = A[r], A[nextSwapped]
        return nextSwapped

      m = partition(A, l, r)
      quickSort(A, l, m - 1)
      quickSort(A, m + 1, r)

    quickSort(nums, 0, len(nums) - 1)
    return nums


# Link: https://leetcode.com/problems/sort-an-array/description/
class Solution:
  def sortArray(self, nums: List[int]) -> List[int]:
    def mergeSort(A: List[int], l: int, r: int) -> None:
      if l >= r:
        return

      def merge(A: List[int], l: int, m: int, r: int) -> None:
        sorted = [0] * (r - l + 1)
        k = 0  # sorted's index
        i = l  # left's index
        j = m + 1  # right's index

        while i <= m and j <= r:
          if A[i] < A[j]:
            sorted[k] = A[i]
            k += 1
            i += 1
          else:
            sorted[k] = A[j]
            k += 1
            j += 1

        # Put the possible remaining left part into the sorted array.
        while i <= m:
          sorted[k] = A[i]
          k += 1
          i += 1

        # Put the possible remaining right part into the sorted array.
        while j <= r:
          sorted[k] = A[j]
          k += 1
          j += 1

        A[l:l + len(sorted)] = sorted

      m = (l + r) // 2
      mergeSort(A, l, m)
      mergeSort(A, m + 1, r)
      merge(A, l, m, r)

    mergeSort(nums, 0, len(nums) - 1)
    return nums


# Link: https://leetcode.com/problems/k-closest-points-to-origin/description/
class Solution:
  def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    def squareDist(p: List[int]) -> int:
      return p[0] * p[0] + p[1] * p[1]

    def quickSelect(l: int, r: int, k: int) -> None:
      pivot = points[r]

      nextSwapped = l
      for i in range(l, r):
        if squareDist(points[i]) <= squareDist(pivot):
          points[nextSwapped], points[i] = points[i], points[nextSwapped]
          nextSwapped += 1
      points[nextSwapped], points[r] = points[r], points[nextSwapped]

      count = nextSwapped - l + 1  the number of points <= pivot
      if count == k:
        return
      if count > k:
        quickSelect(l, nextSwapped - 1, k)
      else:
        quickSelect(nextSwapped + 1, r, k - count)

    quickSelect(0, len(points) - 1, k)
    return points[0:k]


# Link: https://leetcode.com/problems/k-closest-points-to-origin/description/
class Solution:
  def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    maxHeap = []

    for x, y in points:
      heapq.heappush(maxHeap, (- x * x - y * y, [x, y]))
      if len(maxHeap) > k:
        heapq.heappop(maxHeap)

    return [pair[1] for pair in maxHeap]


# Link: https://leetcode.com/problems/k-closest-points-to-origin/description/
class Solution:
  def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    def squareDist(p: List[int]) -> int:
      return p[0] * p[0] + p[1] * p[1]

    def quickSelect(l: int, r: int, k: int) -> None:
      randIndex = random.randint(0, r - l + 1) + l
      points[randIndex], points[r] = points[r], points[randIndex]
      pivot = points[r]

      nextSwapped = l
      for i in range(l, r):
        if squareDist(points[i]) <= squareDist(pivot):
          points[nextSwapped], points[i] = points[i], points[nextSwapped]
          nextSwapped += 1
      points[nextSwapped], points[r] = points[r], points[nextSwapped]

      count = nextSwapped - l + 1  the number of points <= pivot
      if count == k:
        return
      if count > k:
        quickSelect(l, nextSwapped - 1, k)
      else:
        quickSelect(nextSwapped + 1, r, k - count)

    quickSelect(0, len(points) - 1, k)
    return points[0:k]


class Solution:
  def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    def squareDist(p: List[int]) -> int:
      return p[0] * p[0] + p[1] * p[1]

    def quickSelect(l: int, r: int, k: int) -> None:
      pivot = points[r]

      nextSwapped = l
      for i in range(l, r):
        if squareDist(points[i]) <= squareDist(pivot):
          points[nextSwapped], points[i] = points[i], points[nextSwapped]
          nextSwapped += 1
      points[nextSwapped], points[r] = points[r], points[nextSwapped]

      count = nextSwapped - l + 1  the number of points <= pivot
      if count == k:
        return
      if count > k:
        quickSelect(l, nextSwapped - 1, k)
      else:
        quickSelect(nextSwapped + 1, r, k - count)

    quickSelect(0, len(points) - 1, k)
    return points[0:k]


# Link: https://leetcode.com/problems/find-duplicate-subtrees/description/
class Solution:
  def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
    ans = []
    count = collections.Counter()

    def encode(root: Optional[TreeNode]) -> str:
      if not root:
        return ''

      encoded = str(root.val) + '#' + \
          encode(root.left) + '#' + \
          encode(root.right)
      count[encoded] += 1
      if count[encoded] == 2:
        ans.append(root)
      return encoded

    encode(root)
    return ans


# Link: https://leetcode.com/problems/subdomain-visit-count/description/
class Solution:
  def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
    ans = []
    count = collections.Counter()

    for cpdomain in cpdomains:
      num, domains = cpdomain.split()
      num, domains = int(num), domains.split('.')
      for i in reversed(range(len(domains))):
        count['.'.join(domains[i:])] += num

    return [str(freq) + ' ' + domain for domain, freq in count.items()]


# Link: https://leetcode.com/problems/peak-index-in-a-mountain-array/description/
class Solution:
  def peakIndexInMountainArray(self, arr: List[int]) -> int:
    l = 0
    r = len(arr) - 1

    while l < r:
      m = (l + r) // 2
      if arr[m] >= arr[m + 1]:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/happy-students/description/
class Solution:
  def countWays(self, nums: List[int]) -> int:
    return sum(a < i < b
               for i, (a, b) in  # i := the number of the selected numbers
               enumerate(itertools.pairwise([-1] + sorted(nums) + [math.inf])))


# Link: https://leetcode.com/problems/shuffle-an-array/description/
class Solution:
  def __init__(self, nums: List[int]):
    self.nums = nums

  def reset(self) -> List[int]:
    """
    Resets the array to its original configuration and return it.
    """
    return self.nums

  def shuffle(self) -> List[int]:
    """
    Returns a random shuffling of the array.
    """
    A = self.nums.copy()
    for i in range(len(A) - 1, 0, -1):
      j = random.randint(0, i)
      A[i], A[j] = A[j], A[i]
    return A


# Link: https://leetcode.com/problems/adding-spaces-to-a-string/description/
class Solution:
  def addSpaces(self, s: str, spaces: List[int]) -> str:
    ans = []
    j = 0  # spaces' index

    for i, c in enumerate(s):
      if j < len(spaces) and i == spaces[j]:
        ans.append(' ')
        j += 1
      ans.append(c)

    return ''.join(ans)


# Link: https://leetcode.com/problems/partition-string-into-substrings-with-values-at-most-k/description/
class Solution:
  def minimumPartition(self, s: str, k: int) -> int:
    ans = 1
    curr = 0

    for c in s:
      curr = curr * 10 + int(c)
      if curr > k:
        curr = int(c)
        ans += 1
      if curr > k:
        return -1

    return ans


# Link: https://leetcode.com/problems/path-with-minimum-effort/description/
class Solution:
  def minimumEffortPath(self, heights: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(heights)
    n = len(heights[0])
    # diff[i][j] := the maximum absolute difference to reach (i, j)
    diff = [[math.inf] * n for _ in range(m)]
    seen = set()

    minHeap = [(0, 0, 0)]  # (d, i, j)
    diff[0][0] = 0

    while minHeap:
      d, i, j = heapq.heappop(minHeap)
      if i == m - 1 and j == n - 1:
        return d
      seen.add((i, j))
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if (x, y) in seen:
          continue
        newDiff = abs(heights[i][j] - heights[x][y])
        maxDiff = max(diff[i][j], newDiff)
        if diff[x][y] > maxDiff:
          diff[x][y] = maxDiff
          heapq.heappush(minHeap, (diff[x][y], x, y))


# Link: https://leetcode.com/problems/ternary-expression-parser/description/
class Solution:
  def parseTernary(self, expression: str) -> str:
    c = expression[self.i]

    if self.i + 1 == len(expression) or expression[self.i + 1] == ':':
      self.i += 2
      return str(c)

    self.i += 2
    first = self.parseTernary(expression)
    second = self.parseTernary(expression)

    return first if c == 'T' else second

  i = 0


# Link: https://leetcode.com/problems/make-number-of-distinct-characters-equal/description/
class Solution:
  def isItPossible(self, word1: str, word2: str) -> bool:
    count1 = collections.Counter(word1)
    count2 = collections.Counter(word2)
    distinct1 = len(count1)
    distinct2 = len(count2)

    for a in count1:
      for b in count2:
        if a == b:
          # Swapping the same letters won't change the number of distinct
          # letters in each string, so just check if `distinct1 == distinct2`.
          if distinct1 == distinct2:
            return True
          continue
        # The calculation is meaningful only when a != b
        # Swap a in word1 with b in word2.
        distinctAfterSwap1 = distinct1 - (count1[a] == 1) + (count1[b] == 0)
        distinctAfterSwap2 = distinct2 - (count2[b] == 1) + (count2[a] == 0)
        if distinctAfterSwap1 == distinctAfterSwap2:
          return True

    return False


# Link: https://leetcode.com/problems/maximize-number-of-subsequences-in-a-string/description/
class Solution:
  def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
    ans = 0
    count0 = 0  # the count of the letter pattern[0]
    count1 = 0  # the count of the letter pattern[1]

    for c in text:
      if c == pattern[1]:
        ans += count0
        count1 += 1
      if c == pattern[0]:
        count0 += 1

    # It is optimal to add pattern[0] at the beginning or add pattern[1] at the
    # end of the text.
    return ans + max(count0, count1)


# Link: https://leetcode.com/problems/paths-in-maze-that-lead-to-same-room/description/
class Solution:
  def numberOfPaths(self, n: int, corridors: List[List[int]]) -> int:
    ans = 0
    graph = [[False] * 1001 for _ in range(n + 1)]

    for u, v in corridors:
      graph[u][v] = True
      graph[v][u] = True

    for u, v in corridors:
      for i in range(1, n + 1):
        if graph[u][i] and graph[i][v]:
          ans += 1

    return ans // 3


# Link: https://leetcode.com/problems/maximum-number-of-operations-with-the-same-score-ii/description/
class Solution:
  def maxOperations(self, nums: List[int]) -> int:
    @functools.lru_cache(None)
    def dp(i: int, j: int, score: int) -> int:
      """
      Returns the maximum number of operations that can be performed for
      nums[i..j], s.t. all operations have the same `score`.
      """
      if i >= j:
        return 0
      deleteFirstTwo = 1 + dp(i + 2, j, score) \
          if nums[i] + nums[i + 1] == score else 0
      deleteLastTwo = 1 + dp(i, j - 2, score) \
          if nums[j] + nums[j - 1] == score else 0
      deleteFirstAndLast = 1 + dp(i + 1, j - 1, score) \
          if nums[i] + nums[j] == score else 0
      return max(deleteFirstTwo, deleteLastTwo, deleteFirstAndLast)

    n = len(nums)
    return max(dp(0, n - 1, nums[0] + nums[1]),
               dp(0, n - 1, nums[-1] + nums[-2]),
               dp(0, n - 1, nums[0] + nums[-1]))


# Link: https://leetcode.com/problems/design-most-recently-used-queue/description/
from sortedcontainers import SortedList


class MRUQueue:
  def __init__(self, n: int):
    # [(priority value, actual value)]
    self.q = SortedList((i, i) for i in range(1, n + 1))

  def fetch(self, k: int) -> int:
    _, num = self.q.pop(k - 1)
    if self.q:
      maxPriority = self.q[-1][0]
      self.q.add((maxPriority + 1, num))
    else:
      self.q.add((0, num))
    return num


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/description/
class Solution:
  def minOperations(self, nums: List[int], k: int) -> int:
    return functools.reduce(operator.xor, nums, k).bit_count()


# Link: https://leetcode.com/problems/find-original-array-from-doubled-array/description/
class Solution:
  def findOriginalArray(self, changed: List[int]) -> List[int]:
    ans = []
    q = collections.deque()

    for num in sorted(changed):
      if q and num == q[0]:
        q.popleft()
      else:
        q.append(num * 2)
        ans.append(num)

    return [] if q else ans


# Link: https://leetcode.com/problems/stone-game/description/
class Solution:
  def stoneGame(self, piles: List[int]) -> bool:
    n = len(piles)
    # dp[i][j] := the maximum stones you can get more than your opponent in piles[i..j]
    dp = [[0] * n for _ in range(n)]

    for i, pile in enumerate(piles):
      dp[i][i] = pile

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        dp[i][j] = max(piles[i] - dp[i + 1][j],
                       piles[j] - dp[i][j - 1])

    return dp[0][n - 1] > 0


# Link: https://leetcode.com/problems/stone-game/description/
class Solution:
  def stoneGame(self, piles: List[int]) -> bool:
    n = len(piles)
    dp = piles.copy()

    for d in range(1, n):
      for j in range(n - 1, d - 1, -1):
        i = j - d
        dp[j] = max(piles[i] - dp[j], piles[j] - dp[j - 1])

    return dp[n - 1] > 0


# Link: https://leetcode.com/problems/circular-array-loop/description/
class Solution:
  def circularArrayLoop(self, nums: List[int]) -> bool:
    def advance(i: int) -> int:
      return (i + nums[i]) % len(nums)

    if len(nums) < 2:
      return False

    for i, num in enumerate(nums):
      if num == 0:
        continue

      slow = i
      fast = advance(slow)
      while num * nums[fast] > 0 and num * nums[advance(fast)] > 0:
        if slow == fast:
          if slow == advance(slow):
            break
          return True
        slow = advance(slow)
        fast = advance(advance(fast))

      slow = i
      sign = num
      while sign * nums[slow] > 0:
        next = advance(slow)
        nums[slow] = 0
        slow = next

    return False


# Link: https://leetcode.com/problems/count-unguarded-cells-in-the-grid/description/
class Solution:
  def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
    ans = 0
    grid = [[0] * n for _ in range(m)]
    left = [[0] * n for _ in range(m)]
    right = [[0] * n for _ in range(m)]
    up = [[0] * n for _ in range(m)]
    down = [[0] * n for _ in range(m)]

    for row, col in guards:
      grid[row][col] = 'G'

    for row, col in walls:
      grid[row][col] = 'W'

    for i in range(m):
      lastCell = 0
      for j in range(n):
        if grid[i][j] == 'G' or grid[i][j] == 'W':
          lastCell = grid[i][j]
        else:
          left[i][j] = lastCell
      lastCell = 0
      for j in range(n - 1, -1, -1):
        if grid[i][j] == 'G' or grid[i][j] == 'W':
          lastCell = grid[i][j]
        else:
          right[i][j] = lastCell

    for j in range(n):
      lastCell = 0
      for i in range(m):
        if grid[i][j] == 'G' or grid[i][j] == 'W':
          lastCell = grid[i][j]
        else:
          up[i][j] = lastCell
      lastCell = 0
      for i in range(m - 1, -1, -1):
        if grid[i][j] == 'G' or grid[i][j] == 'W':
          lastCell = grid[i][j]
        else:
          down[i][j] = lastCell

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 0 and left[i][j] != 'G' and right[i][j] != 'G' and \
                up[i][j] != 'G' and down[i][j] != 'G':
          ans += 1

    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-make-array-equal-ii/description/
class Solution:
  def minOperations(self, nums1: List[int], nums2: List[int], k: int) -> int:
    if k == 0:
      return 0 if nums1 == nums2 else -1

    ans = 0
    opsDiff = 0  # the number of increments - number of decrements

    for num1, num2 in zip(nums1, nums2):
      diff = num1 - num2
      if diff == 0:
        continue
      if diff % k != 0:
        return -1
      ops = diff // k
      opsDiff += ops
      ans += abs(ops)

    return ans // 2 if opsDiff == 0 else -1


# Link: https://leetcode.com/problems/maximum-number-of-moves-in-a-grid/description/
class Solution:
  def maxMoves(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    # dp[i][j] := the maximum number of moves you can perform from (i, j)
    dp = [[0] * n for _ in range(m)]

    for j in range(n - 2, -1, -1):
      for i in range(m):
        if grid[i][j + 1] > grid[i][j]:
          dp[i][j] = 1 + dp[i][j + 1]
        if i > 0 and grid[i - 1][j + 1] > grid[i][j]:
          dp[i][j] = max(dp[i][j], 1 + dp[i - 1][j + 1])
        if i + 1 < m and grid[i + 1][j + 1] > grid[i][j]:
          dp[i][j] = max(dp[i][j], 1 + dp[i + 1][j + 1])

    return max(dp[i][0] for i in range(m))


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root.val > max(p.val, q.val):
      return self.lowestCommonAncestor(root.left, p, q)
    if root.val < min(p.val, q.val):
      return self.lowestCommonAncestor(root.right, p, q)
    return root


# Link: https://leetcode.com/problems/sum-of-even-numbers-after-queries/description/
class Solution:
  def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    ans = []
    summ = sum(a for a in nums if a % 2 == 0)

    for val, index in queries:
      if nums[index] % 2 == 0:
        summ -= nums[index]
      nums[index] += val
      if nums[index] % 2 == 0:
        summ += nums[index]
      ans.append(summ)

    return ans


# Link: https://leetcode.com/problems/find-three-consecutive-integers-that-sum-to-a-given-number/description/
class Solution:
  def sumOfThree(self, num: int) -> List[int]:
    if num % 3:
      return []
    x = num // 3
    return [x - 1, x, x + 1]


# Link: https://leetcode.com/problems/divide-two-integers/description/
class Solution:
  def divide(self, dividend: int, divisor: int) -> int:
    # -2^{31} / -1 = 2^31 will overflow, so return 2^31 - 1.
    if dividend == -2**31 and divisor == -1:
      return 2**31 - 1

    sign = -1 if (dividend > 0) ^ (divisor > 0) else 1
    ans = 0
    dvd = abs(dividend)
    dvs = abs(divisor)

    while dvd >= dvs:
      k = 1
      while k * 2 * dvs <= dvd:
        k <<= 1
      dvd -= k * dvs
      ans += k

    return sign * ans


# Link: https://leetcode.com/problems/find-the-divisibility-array-of-a-string/description/
class Solution:
  def divisibilityArray(self, word: str, m: int) -> List[int]:
    ans = []
    prevRemainder = 0

    for c in word:
      remainder = (prevRemainder * 10 + int(c)) % m
      ans.append(1 if remainder == 0 else 0)
      prevRemainder = remainder

    return ans


# Link: https://leetcode.com/problems/integer-to-roman/description/
class Solution:
  def intToRoman(self, num: int) -> str:
    M = ['', 'M', 'MM', 'MMM']
    C = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
    X = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
    I = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
    return M[num // 1000] + C[num % 1000 // 100] + X[num % 100 // 10] + I[num % 10]


# Link: https://leetcode.com/problems/integer-to-roman/description/
class Solution:
  def intToRoman(self, num: int) -> str:
    valueSymbols = [(1000, 'M'), (900, 'CM'),
                    (500, 'D'), (400, 'CD'),
                    (100, 'C'), (90, 'XC'),
                    (50, 'L'), (40, 'XL'),
                    (10, 'X'), (9, 'IX'),
                    (5, 'V'), (4, 'IV'),
                    (1, 'I')]
    ans = []

    for value, symbol in valueSymbols:
      if num == 0:
        break
      count, num = divmod(num, value)
      ans.append(symbol * count)

    return ''.join(ans)


# Link: https://leetcode.com/problems/daily-temperatures/description/
class Solution:
  def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    ans = [0] * len(temperatures)
    stack = []  # a decreasing stack

    for i, temperature in enumerate(temperatures):
      while stack and temperature > temperatures[stack[-1]]:
        index = stack.pop()
        ans[index] = i - index
      stack.append(i)

    return ans


# Link: https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips/description/
class Solution:
  def removeOnes(self, grid: List[List[int]]) -> bool:
    revRow = [a ^ 1 for a in grid[0]]
    return all(row == grid[0] or row == revRow for row in grid)


# Link: https://leetcode.com/problems/combination-sum/description/
class Solution:
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    ans = []

    def dfs(s: int, target: int, path: List[int]) -> None:
      if target < 0:
        return
      if target == 0:
        ans.append(path.clone())
        return

      for i in range(s, len(candidates)):
        path.append(candidates[i])
        dfs(i, target - candidates[i], path)
        path.pop()

    candidates.sort()
    dfs(0, target, [])
    return ans


# Link: https://leetcode.com/problems/most-beautiful-item-for-each-query/description/
class Solution:
  def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
    prices, beauties = zip(*sorted(items))
    maxBeautySoFar = [0] * (len(beauties) + 1)

    for i, beauty in enumerate(beauties):
      maxBeautySoFar[i + 1] = max(maxBeautySoFar[i], beauty)

    return [maxBeautySoFar[bisect_right(prices, query)] for query in queries]


# Link: https://leetcode.com/problems/group-the-people-given-the-group-size-they-belong-to/description/
class Solution:
  def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
    ans = []
    groupSizeToIndices = defaultdict(list)

    for i, groupSize in enumerate(groupSizes):
      groupSizeToIndices[groupSize].append(i)

    for groupSize, indices in groupSizeToIndices.items():
      groupIndices = []
      for index in indices:
        groupIndices.append(index)
        if len(groupIndices) == groupSize:
          ans.append(groupIndices.copy())
          groupIndices.clear()

    return ans


# Link: https://leetcode.com/problems/count-substrings-that-differ-by-one-character/description/
class Solution:
  def countSubstrings(self, s: str, t: str) -> int:
    ans = 0

    for i in range(len(s)):
      ans += self._count(s, t, i, 0)

    for j in range(1, len(t)):
      ans += self._count(s, t, 0, j)

    return ans

  def _count(self, s: str, t: str, i: int, j: int) -> int:
    """Returns the number of substrings of s[i..n) and t[j:] that differ by one char."""
    res = 0
    # the number of substrings starting at s[i] and t[j] ending in the current
    # index with zero different letter
    dp0 = 0
    # the number of substrings starting at s[i] and t[j] ending in the current
    # index with one different letter
    dp1 = 0

    while i < len(s) and j < len(t):
      if s[i] == t[j]:
        dp0 += 1
      else:
        dp0, dp1 = 0, dp0 + 1
      res += dp1
      i += 1
      j += 1

    return res


# Link: https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows/description/
class Solution:
  def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
    patterns = [tuple(a ^ row[0] for a in row) for row in matrix]
    return max(Counter(patterns).values())


# Link: https://leetcode.com/problems/maximum-gap/description/
class Bucket:
  def __init__(self, mini: int, maxi: int):
    self.mini = mini
    self.maxi = maxi


class Solution:
  def maximumGap(self, nums: List[int]) -> int:
    if len(nums) < 2:
      return 0

    mini = min(nums)
    maxi = max(nums)
    if mini == maxi:
      return 0

    gap = math.ceil((maxi - mini) / (len(nums) - 1))
    bucketSize = (maxi - mini) // gap + 1
    buckets = [Bucket(math.inf, -math.inf) for _ in range(bucketSize)]

    for num in nums:
      i = (num - mini) // gap
      buckets[i].mini = min(buckets[i].mini, num)
      buckets[i].maxi = max(buckets[i].maxi, num)

    ans = 0
    prevMax = mini

    for bucket in buckets:
      if bucket.mini == math.inf:
        continue  # empty bucket
      ans = max(ans, bucket.mini - prevMax)
      prevMax = bucket.maxi

    return ans


# Link: https://leetcode.com/problems/maximize-greatness-of-an-array/description/
class Solution:
  def maximizeGreatness(self, nums: List[int]) -> int:
    ans = 0

    nums.sort()

    for num in nums:
      if num > nums[ans]:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/valid-parenthesis-string/description/
class Solution:
  def checkValidString(self, s: str) -> bool:
    low = 0
    high = 0

    for c in s:
      if c == '(':
        low += 1
        high += 1
      elif c == ')':
        if low > 0:
          low -= 1
        high -= 1
      else:
        if low > 0:
          low -= 1
        high += 1
      if high < 0:
        return False

    return low == 0


# Link: https://leetcode.com/problems/count-submatrices-with-top-left-element-and-sum-less-than-k/description/
class Solution:
  def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
    m = len(grid)
    n = len(grid[0])
    ans = 0
    # prefix[i][j] := the sum of matrix[0..i)[0..j)
    prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        prefix[i + 1][j + 1] = \
            grid[i][j] + prefix[i][j + 1] + \
            prefix[i + 1][j] - prefix[i][j]
        if prefix[i + 1][j + 1] <= k:
          ans += 1

    return ans


# Link: https://leetcode.com/problems/insert-into-a-binary-search-tree/description/
class Solution:
  def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if not root:
      return TreeNode(val)
    if root.val > val:
      root.left = self.insertIntoBST(root.left, val)
    else:
      root.right = self.insertIntoBST(root.right, val)
    return root


# Link: https://leetcode.com/problems/walls-and-gates/description/
class Solution:
  def wallsAndGates(self, rooms: List[List[int]]) -> None:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    kInf = 2**31 - 1
    m = len(rooms)
    n = len(rooms[0])
    q = collections.deque()

    for i in range(m):
      for j in range(n):
        if rooms[i][j] == 0:
          q.append((i, j))

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if rooms[x][y] != kInf:
          continue
        rooms[x][y] = rooms[i][j] + 1
        q.append((x, y))


# Link: https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/description/
class Solution:
  def longestConsecutive(self, root: Optional[TreeNode]) -> int:
    if not root:
      return 0

    def dfs(root: Optional[TreeNode], target: int, length: int, maxLength: int) -> int:
      if not root:
        return maxLength
      if root.val == target:
        length += 1
        maxLength = max(maxLength, length)
      else:
        length = 1
      return max(dfs(root.left, root.val + 1, length, maxLength),
                 dfs(root.right, root.val + 1, length, maxLength))

    return dfs(root, root.val, 0, 0)


# Link: https://leetcode.com/problems/pyramid-transition-matrix/description/
class Solution:
  def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
    prefixToBlocks = collections.defaultdict(list)

    for a in allowed:
      prefixToBlocks[a[:2]].append(a[2])

    def dfs(row: str, nextRow: str, i: int) -> bool:
      if len(row) == 1:
        return True
      if len(nextRow) + 1 == len(row):
        return dfs(nextRow, '', 0)

      for c in prefixToBlocks[row[i:i + 2]]:
        if dfs(row, nextRow + c, i + 1):
          return True

      return False

    return dfs(bottom, '', 0)


# Link: https://leetcode.com/problems/kill-process/description/
class Solution:
  def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
    ans = []
    tree = collections.defaultdict(list)

    for v, u in zip(pid, ppid):
      if u == 0:
        continue
      tree[u].append(v)

    def dfs(u: int) -> None:
      ans.append(u)
      for v in tree.get(u, []):
        dfs(v)

    dfs(kill)
    return ans


# Link: https://leetcode.com/problems/maximum-length-of-pair-chain/description/
class Solution:
  def findLongestChain(self, pairs: List[List[int]]) -> int:
    ans = 0
    prevEnd = -math.inf

    for s, e in sorted(pairs, key=lambda x: x[1]):
      if s > prevEnd:
        ans += 1
        prevEnd = e

    return ans


# Link: https://leetcode.com/problems/number-of-ways-to-buy-pens-and-pencils/description/
class Solution:
  def waysToBuyPensPencils(self, total: int, cost1: int, cost2: int) -> int:
    maxPen = total // cost1
    return sum((total - i * cost1) // cost2
               for i in range(maxPen + 1)) + maxPen + 1


# Link: https://leetcode.com/problems/minimum-addition-to-make-integer-beautiful/description/
class Solution:
  def makeIntegerBeautiful(self, n: int, target: int) -> int:
    ans = 0
    power = 1

    # e.g. n = 123. After tunning off the last bit by adding 7, n = 130.
    # Effectively, we can think n as 13. That's why we do n = (n / 10) + 1.
    while sum(map(int, str(n))) > target:
      # the cost to turn off the last digit
      ans += power * (10 - n % 10)
      n = n // 10 + 1
      power *= 10

    return ans


# Link: https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/description/
class Solution:
  def minimumLength(self, s: str) -> int:
    i = 0
    j = len(s) - 1

    while i < j and s[i] == s[j]:
      c = s[i]
      while i <= j and s[i] == c:
        i += 1
      while i <= j and s[j] == c:
        j -= 1

    return j - i + 1


# Link: https://leetcode.com/problems/find-and-replace-pattern/description/
class Solution:
  def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
    def isIsomorphic(w: str, p: str) -> bool:
      return [*map(w.index, w)] == [*map(p.index, p)]
    return [word for word in words if isIsomorphic(word, pattern)]


# Link: https://leetcode.com/problems/simple-bank-system/description/
class Bank:
  def __init__(self, balance: List[int]):
    self.balance = balance

  def transfer(self, account1: int, account2: int, money: int) -> bool:
    if not self._isValid(account2):
      return False
    return self.withdraw(account1, money) and self.deposit(account2, money)

  def deposit(self, account: int, money: int) -> bool:
    if not self._isValid(account):
      return False
    self.balance[account - 1] += money
    return True

  def withdraw(self, account: int, money: int) -> bool:
    if not self._isValid(account):
      return False
    if self.balance[account - 1] < money:
      return False
    self.balance[account - 1] -= money
    return True

  def _isValid(self, account: int) -> bool:
    return 1 <= account <= len(self.balance)


# Link: https://leetcode.com/problems/wiggle-subsequence/description/
class Solution:
  def wiggleMaxLength(self, nums: List[int]) -> int:
    increasing = 1
    decreasing = 1

    for a, b in itertools.pairwise(nums):
      if b > a:
        increasing = decreasing + 1
      elif b < a:
        decreasing = increasing + 1

    return max(increasing, decreasing)


# Link: https://leetcode.com/problems/snapshot-array/description/
class SnapshotArray:
  def __init__(self, length: int):
    self.snaps = [[[0, 0]] for _ in range(length)]
    self.snap_id = 0

  def set(self, index: int, val: int) -> None:
    snap = self.snaps[index][-1]
    if snap[0] == self.snap_id:
      snap[1] = val
    else:
      self.snaps[index].append([self.snap_id, val])

  def snap(self) -> int:
    self.snap_id += 1
    return self.snap_id - 1

  def get(self, index: int, snap_id: int) -> int:
    i = bisect_left(self.snaps[index], [snap_id + 1]) - 1
    return self.snaps[index][i][1]


# Link: https://leetcode.com/problems/count-the-number-of-good-subsequences/description/
class Solution:
  def countGoodSubsequences(self, s: str) -> int:
    kMod = 1_000_000_007
    ans = 0
    count = collections.Counter(s)

    @functools.lru_cache(None)
    def fact(i: int) -> int:
      return 1 if i <= 1 else i * fact(i - 1) % kMod

    @functools.lru_cache(None)
    def inv(i: int) -> int:
      return pow(i, kMod - 2, kMod)

    @functools.lru_cache(None)
    def nCk(n: int, k: int) -> int:
      return fact(n) * inv(fact(k)) * inv(fact(n - k)) % kMod

    for freq in range(1, max(count.values()) + 1):
      numSubseqs = 1  # ""
      for charFreq in count.values():
        if charFreq >= freq:
          numSubseqs = numSubseqs * (1 + nCk(charFreq, freq)) % kMod
      ans += numSubseqs - 1  # Minus "".
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/moving-stones-until-consecutive-ii/description/
class Solution:
  def numMovesStonesII(self, stones: List[int]) -> List[int]:
    n = len(stones)
    minMoves = n

    stones.sort()

    l = 0
    for r, stone in enumerate(stones):
      while stone - stones[l] + 1 > n:
        l += 1
      alreadyStored = r - l + 1
      if alreadyStored == n - 1 and stone - stones[l] + 1 == n - 1:
        minMoves = 2
      else:
        minMoves = min(minMoves, n - alreadyStored)

    return [minMoves, max(stones[n - 1] - stones[1] - n + 2, stones[n - 2] - stones[0] - n + 2)]


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root or root == p or root == q:
      return root
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    if left and right:
      return root
    return left or right


# Link: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/
class Solution:
  def lowestCommonAncestor(self, root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    q_ = collections.deque([root])
    parent = {root: None}
    ancestors = set()  # p's ancestors

    # Iterate until we find both p and q.
    while p not in parent or q not in parent:
      root = q_.popleft()
      if root.left:
        parent[root.left] = root
        q_.append(root.left)
      if root.right:
        parent[root.right] = root
        q_.append(root.right)

    # Insert all the p's ancestors.
    while p:
      ancestors.add(p)
      p = parent[p]  # `p` becomes None in the end.

    # Go up from q until we meet any of p's ancestors.
    while q not in ancestors:
      q = parent[q]

    return q


# Link: https://leetcode.com/problems/design-a-todo-list/description/
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
  taskDescription: str
  dueDate: int
  tags: List[str]


class TodoList:
  def __init__(self):
    self.taskId = 0
    self.taskIds = set()
    self.userIdToTaskIdToTasks: Dict[int, Dict[int, List[Task]]] = {}

  def addTask(self, userId: int, taskDescription: str, dueDate: int, tags: List[str]) -> int:
    self.taskId += 1
    taskIdToTasks = self.userIdToTaskIdToTasks.setdefault(userId, {})
    taskIdToTasks[self.taskId] = Task(taskDescription, dueDate, tags)
    self.taskIds.add(self.taskId)
    return self.taskId

  def getAllTasks(self, userId: int) -> List[str]:
    return [task.taskDescription
            for task in self._getTasksSortedByDueDate(userId)]

  def getTasksForTag(self, userId: int, tag: str) -> List[str]:
    return [task.taskDescription
            for task in self._getTasksSortedByDueDate(userId)
            if tag in task.tags]

  def completeTask(self, userId: int, taskId: int) -> None:
    if taskId not in self.taskIds:
      return
    if userId not in self.userIdToTaskIdToTasks:
      return
    taskIdToTasks = self.userIdToTaskIdToTasks[userId]
    if taskId not in taskIdToTasks:
      return
    del taskIdToTasks[taskId]

  def _getTasksSortedByDueDate(self, userId: int) -> List[Task]:
    if userId not in self.userIdToTaskIdToTasks:
      return []
    taskIdToTasks = self.userIdToTaskIdToTasks[userId]
    return sorted([task for task in taskIdToTasks.values()], key=lambda x: x.dueDate)


# Link: https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/description/
class Solution:
  def concatenatedBinary(self, n: int) -> int:
    kMod = 1_000_000_007
    ans = 0

    def numberOfBits(n: int) -> int:
      return int(math.log2(n)) + 1

    for i in range(1, n + 1):
      ans = ((ans << numberOfBits(i)) + i) % kMod

    return ans


# Link: https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/description/
class Solution:
  def concatenatedBinary(self, n: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    numberOfBits = 0

    for i in range(1, n + 1):
      if i.bit_count() == 1:
        numberOfBits += 1
      ans = ((ans << numberOfBits) + i) % kMod

    return ans


# Link: https://leetcode.com/problems/course-schedule-ii/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    ans = []
    graph = [[] for _ in range(numCourses)]
    states = [State.kInit] * numCourses

    for v, u in prerequisites:
      graph[u].append(v)

    def hasCycle(u: int) -> bool:
      if states[u] == State.kVisiting:
        return True
      if states[u] == State.kVisited:
        return False

      states[u] = State.kVisiting
      if any(hasCycle(v) for v in graph[u]):
        return True
      states[u] = State.kVisited
      ans.append(u)

      return False

    if any(hasCycle(i) for i in range(numCourses)):
      return []

    return ans[::-1]


# Link: https://leetcode.com/problems/course-schedule-ii/description/
class Solution:
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    ans = []
    graph = [[] for _ in range(numCourses)]
    inDegrees = [0] * numCourses
    q = collections.deque()

    # Build the graph.
    for v, u in prerequisites:
      graph[u].append(v)
      inDegrees[v] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    while q:
      u = q.popleft()
      ans.append(u)
      for v in graph[u]:
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return ans if len(ans) == numCourses else []


# Link: https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/description/
class Solution:
  def findTheLongestSubstring(self, s: str) -> int:
    kVowels = 'aeiou'
    ans = 0
    prefix = 0  # the binary prefix
    prefixToIndex = {0: -1}

    for i, c in enumerate(s):
      index = kVowels.find(c)
      if index != -1:
        prefix ^= 1 << index
      prefixToIndex.setdefault(prefix, i)
      ans = max(ans, i - prefixToIndex[prefix])

    return ans


# Link: https://leetcode.com/problems/maximal-network-rank/description/
class Solution:
  def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
    degrees = [0] * n

    for u, v in roads:
      degrees[u] += 1
      degrees[v] += 1

    # Find the first maximum and the second maximum degrees.
    maxDegree1 = 0
    maxDegree2 = 0
    for degree in degrees:
      if degree > maxDegree1:
        maxDegree2 = maxDegree1
        maxDegree1 = degree
      elif degree > maxDegree2:
        maxDegree2 = degree

    # There can be multiple nodes with `maxDegree1` or `maxDegree2`.
    # Find the counts of such nodes.
    countMaxDegree1 = 0
    countMaxDegree2 = 0
    for degree in degrees:
      if degree == maxDegree1:
        countMaxDegree1 += 1
      elif degree == maxDegree2:
        countMaxDegree2 += 1

    if countMaxDegree1 == 1:
      # Case 1: If there is only one node with degree = `maxDegree1`, then
      # we'll need to use the node with degree = `maxDegree2`. The answer in
      # general will be (maxDegree1 + maxDegree2), but if the two nodes that
      # we're considering are connected, then we'll have to subtract 1.
      edgeCount = self._getEdgeCount(roads, degrees, maxDegree1, maxDegree2) + \
          self._getEdgeCount(roads, degrees, maxDegree2, maxDegree1)
      return maxDegree1 + maxDegree2 - (countMaxDegree2 == edgeCount)
    else:
      # Case 2: If there are more than one node with degree = `maxDegree1`,
      # then we can consider `maxDegree1` twice, and we don't need to use
      # `maxDegree2`. The answer in general will be 2 * maxDegree1, but if the
      # two nodes that we're considering are connected, then we'll have to
      # subtract 1.
      edgeCount = self._getEdgeCount(roads, degrees, maxDegree1, maxDegree1)
      maxPossibleEdgeCount = countMaxDegree1 * (countMaxDegree1 - 1) // 2
      return 2 * maxDegree1 - (maxPossibleEdgeCount == edgeCount)

  def _getEdgeCount(self, roads: List[List[int]], degrees: List[int], degreeU: int, degreeV: int) -> int:
    """
    Returns the number of edges (u, v) where degress[u] == degreeU and
    degrees[v] == degreeV.
    """
    edgeCount = 0
    for u, v in roads:
      if degrees[u] == degreeU and degrees[v] == degreeV:
        edgeCount += 1
    return edgeCount


# Link: https://leetcode.com/problems/optimal-division/description/
class Solution:
  def optimalDivision(self, nums: List[int]) -> str:
    ans = str(nums[0])

    if len(nums) == 1:
      return ans
    if len(nums) == 2:
      return ans + '/' + str(nums[1])

    ans += '/(' + str(nums[1])
    for i in range(2, len(nums)):
      ans += '/' + str(nums[i])
    ans += ')'
    return ans


# Link: https://leetcode.com/problems/design-a-food-rating-system/description/
from sortedcontainers import SortedSet


class FoodRatings:
  def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
    self.cuisineToRatingAndFoods = collections.defaultdict(
        lambda: SortedSet(key=lambda x: (-x[0], x[1])))
    self.foodToCuisine = {}
    self.foodToRating = {}

    for food, cuisine, rating in zip(foods, cuisines, ratings):
      self.cuisineToRatingAndFoods[cuisine].add((rating, food))
      self.foodToCuisine[food] = cuisine
      self.foodToRating[food] = rating

  def changeRating(self, food: str, newRating: int) -> None:
    cuisine = self.foodToCuisine[food]
    oldRating = self.foodToRating[food]
    ratingAndFoods = self.cuisineToRatingAndFoods[cuisine]
    ratingAndFoods.remove((oldRating, food))
    ratingAndFoods.add((newRating, food))
    self.foodToRating[food] = newRating

  def highestRated(self, cuisine: str) -> str:
    return self.cuisineToRatingAndFoods[cuisine][0][1]


# Link: https://leetcode.com/problems/sum-game/description/
class Solution:
  def sumGame(self, num: str) -> bool:
    n = len(num)
    ans = 0.0

    def getExpectation(c: str) -> float:
      return 4.5 if c == '?' else ord(c) - ord('0')

    for i in range(n // 2):
      ans += getExpectation(num[i])

    for i in range(n // 2, n):
      ans -= getExpectation(num[i])

    return ans != 0.0


# Link: https://leetcode.com/problems/partition-array-for-maximum-sum/description/
class Solution:
  def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
    n = len(arr)
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
      maxi = -math.inf
      for j in range(1, min(i, k) + 1):
        maxi = max(maxi, arr[i - j])
        dp[i] = max(dp[i], dp[i - j] + maxi * j)

    return dp[n]


# Link: https://leetcode.com/problems/the-knights-tour/description/
class Solution:
  def tourOfKnight(self, m: int, n: int, r: int, c: int) -> List[List[int]]:
    dirs = ((1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    ans = [[-1] * n for _ in range(m)]

    def dfs(i: int, j: int, step: int) -> bool:
      if step == m * n:
        return True
      if i < 0 or i >= m or j < 0 or j >= n:
        return False
      if ans[i][j] != -1:
        return False
      ans[i][j] = step
      for dx, dy in dirs:
        if dfs(i + dx, j + dy, step + 1):
          return True
      ans[i][j] = -1
      return False

    dfs(r, c, 0)
    return ans


# Link: https://leetcode.com/problems/minimum-time-to-complete-trips/description/
class Solution:
  def minimumTime(self, time: List[int], totalTrips: int) -> int:
    l = 1
    r = min(time) * totalTrips

    while l < r:
      m = (l + r) // 2
      if sum(m // t for t in time) >= totalTrips:
        r = m
      else:
        l = m + 1

    return l


# Link: https://leetcode.com/problems/matchsticks-to-square/description/
class Solution:
  def makesquare(self, matchsticks: List[int]) -> bool:
    if len(matchsticks) < 4:
      return False

    perimeter = sum(matchsticks)
    if perimeter % 4 != 0:
      return False

    A = sorted(matchsticks)[::-1]

    def dfs(selected: int, edges: List[int]) -> bool:
      if selected == len(A):
        return all(edge == edges[0] for edge in edges)

      for i, edge in enumerate(edges):
        if A[selected] > edge:
          continue
        edges[i] -= A[selected]
        if dfs(selected + 1, edges):
          return True
        edges[i] += A[selected]

      return False

    return dfs(0, [perimeter // 4] * 4)


# Link: https://leetcode.com/problems/sum-of-mutated-array-closest-to-target/description/
class Solution:
  def findBestValue(self, arr: List[int], target: int) -> int:
    prefix = 0

    arr.sort()

    for i, a in enumerate(arr):
      ans = round((target - prefix) / (len(arr) - i))
      if ans <= a:
        return ans
      prefix += a

    return arr[-1]


# Link: https://leetcode.com/problems/check-if-a-string-can-break-another-string/description/
class Solution:
  def checkIfCanBreak(self, s1: str, s2: str) -> bool:
    count = collections.Counter(s1)
    count.subtract(collections.Counter(s2))

    for a, b in itertools.pairwise(string.ascii_lowercase):
      count[b] += count[a]

    return all(value <= 0 for value in count.values()) or \
        all(value >= 0 for value in count.values())


# Link: https://leetcode.com/problems/check-if-a-string-can-break-another-string/description/
class Solution:
  def checkIfCanBreak(self, s1: str, s2: str) -> bool:
    count1 = collections.Counter(s1)
    count2 = collections.Counter(s2)

    def canBreak(count1: Dict[str, int], count2: Dict[str, int]) -> bool:
      """Returns True if count1 can break count2."""
      diff = 0
      for c in string.ascii_lowercase:
        diff += count2[c] - count1[c]
        # count2 is alphabetically greater than count1.
        if diff < 0:
          return False
      return True

    return canBreak(count1, count2) or canBreak(count2, count1)


# Link: https://leetcode.com/problems/factor-combinations/description/
class Solution:
  def getFactors(self, n: int) -> List[List[int]]:
    ans = []

    def dfs(n: int, s: int, path: List[int]) -> None:
      if n <= 1:
        if len(path) > 1:
          ans.append(path.copy())
        return

      for i in range(s, n + 1):
        if n % i == 0:
          path.append(i)
          dfs(n // i, i, path)
          path.pop()

    dfs(n, 2, [])  # The minimum factor is 2.
    return ans


# Link: https://leetcode.com/problems/combination-sum-iv/description/
class Solution:
  def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [1] + [-1] * target

    def dfs(target: int) -> int:
      if target < 0:
        return 0
      if dp[target] != -1:
        return dp[target]

      dp[target] = sum(dfs(target - num) for num in nums)
      return dp[target]

    return dfs(target)


# Link: https://leetcode.com/problems/find-the-closest-marked-node/description/
class Solution:
  def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
      graph[u].append((v, w))

    dist = self._dijkstra(graph, s)
    ans = min(dist[u] for u in marked)
    return -1 if ans == math.inf else ans

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int) -> List[int]:
    dist = [math.inf] * len(graph)

    dist[src] = 0
    minHeap = [(dist[src], src)]  # (d, u)

    while minHeap:
      d, u = heapq.heappop(minHeap)
      for v, w in graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (dist[v], v))

    return dist


# Link: https://leetcode.com/problems/angle-between-hands-of-a-clock/description/
class Solution:
  def angleClock(self, hour: int, minutes: int) -> float:
    hourAngle = (hour % 12) * 30 + minutes * 0.5
    minuteAngle = minutes * 6
    ans = abs(hourAngle - minuteAngle)

    return min(ans, 360 - ans)


# Link: https://leetcode.com/problems/largest-combination-with-bitwise-and-greater-than-zero/description/
class Solution:
  def largestCombination(self, candidates: List[int]) -> int:
    return max(sum(c >> i & 1 for c in candidates) for i in range(24))


# Link: https://leetcode.com/problems/count-number-of-teams/description/
class Solution:
  def numTeams(self, rating: List[int]) -> int:
    ans = 0

    for i in range(1, len(rating) - 1):
      # Calculate soldiers on the left with less//greater ratings.
      leftLess = 0
      leftGreater = 0
      for j in range(i):
        if rating[j] < rating[i]:
          leftLess += 1
        elif rating[j] > rating[i]:
          leftGreater += 1
      # Calculate soldiers on the right with less//greater ratings.
      rightLess = 0
      rightGreater = 0
      for j in range(i + 1, len(rating)):
        if rating[j] < rating[i]:
          rightLess += 1
        elif rating[j] > rating[i]:
          rightGreater += 1
      ans += leftLess * rightGreater + leftGreater * rightLess

    return ans


# Link: https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/description/
class Solution:
  def minSum(self, nums1: List[int], nums2: List[int]) -> int:
    sum1 = sum(nums1)
    sum2 = sum(nums2)
    zero1 = nums1.count(0)
    zero2 = nums2.count(0)
    if zero1 == 0 and sum1 < sum2 + zero2:
      return -1
    if zero2 == 0 and sum2 < sum1 + zero1:
      return -1
    return max(sum1 + zero1, sum2 + zero2)


# Link: https://leetcode.com/problems/count-total-number-of-colored-cells/description/
class Solution:
  def coloredCells(self, n: int) -> int:
    return n**2 + (n - 1)**2


# Link: https://leetcode.com/problems/pacific-atlantic-water-flow/description/
class Solution:
  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    m = len(heights)
    n = len(heights[0])
    seenP = [[False] * n for _ in range(m)]
    seenA = [[False] * n for _ in range(m)]

    def dfs(i: int, j: int, h: int, seen: List[List[bool]]) -> None:
      if i < 0 or i == m or j < 0 or j == n:
        return
      if seen[i][j] or heights[i][j] < h:
        return

      seen[i][j] = True
      dfs(i + 1, j, heights[i][j], seen)
      dfs(i - 1, j, heights[i][j], seen)
      dfs(i, j + 1, heights[i][j], seen)
      dfs(i, j - 1, heights[i][j], seen)

    for i in range(m):
      dfs(i, 0, 0, seenP)
      dfs(i, n - 1, 0, seenA)

    for j in range(n):
      dfs(0, j, 0, seenP)
      dfs(m - 1, j, 0, seenA)

    return [[i, j]
            for i in range(m)
            for j in range(n)
            if seenP[i][j] and seenA[i][j]]


# Link: https://leetcode.com/problems/pacific-atlantic-water-flow/description/
class Solution:
  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(heights)
    n = len(heights[0])
    qP = collections.deque()
    qA = collections.deque()
    seenP = [[False] * n for _ in range(m)]
    seenA = [[False] * n for _ in range(m)]

    for i in range(m):
      qP.append((i, 0))
      qA.append((i, n - 1))
      seenP[i][0] = True
      seenA[i][n - 1] = True

    for j in range(n):
      qP.append((0, j))
      qA.append((m - 1, j))
      seenP[0][j] = True
      seenA[m - 1][j] = True

    def bfs(q: deque, seen: List[List[bool]]):
      while q:
        i, j = q.popleft()
        h = heights[i][j]
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if seen[x][y] or heights[x][y] < h:
            continue
          q.append((x, y))
          seen[x][y] = True

    bfs(qP, seenP)
    bfs(qA, seenA)

    return [[i, j] for i in range(m) for j in range(n) if seenP[i][j] and seenA[i][j]]


# Link: https://leetcode.com/problems/longest-word-in-dictionary-through-deleting/description/
class Solution:
  def findLongestWord(self, s: str, d: List[str]) -> str:
    ans = ''

    for word in d:
      i = 0
      for c in s:
        if i < len(word) and c == word[i]:
          i += 1
      if i == len(word):
        if len(word) > len(ans) or len(word) == len(ans) and word < ans:
          ans = word

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-eaten-apples/description/
class Solution:
  def eatenApples(self, apples: List[int], days: List[int]) -> int:
    n = len(apples)
    ans = 0
    minHeap = []  # (the rotten day, the number of apples)

    i = 0
    while i < n or minHeap:
      # Remove the rotten apples.
      while minHeap and minHeap[0][0] <= i:
        heapq.heappop(minHeap)
      # Add today's apples.
      if i < n and apples[i] > 0:
        heapq.heappush(minHeap, (i + days[i], apples[i]))
      # Eat one apple today.
      if minHeap:
        rottenDay, numApples = heapq.heappop(minHeap)
        if numApples > 1:
          heapq.heappush(minHeap, (rottenDay, numApples - 1))
        ans += 1
      i += 1

    return ans


# Link: https://leetcode.com/problems/minimum-area-rectangle-ii/description/
class Solution:
  def minAreaFreeRect(self, points: List[List[int]]) -> float:
    ans = math.inf
    # For each A, B pair points, {hash(A, B): (ax, ay, bx, by)}.
    centerToPoints = collections.defaultdict(list)

    for ax, ay in points:
      for bx, by in points:
        center = ((ax + bx) / 2, (ay + by) / 2)
        centerToPoints[center].append((ax, ay, bx, by))

    def dist(px: int, py: int, qx: int, qy: int) -> float:
      return (px - qx)**2 + (py - qy)**2

    # For all pair points "that share the same center".
    for points in centerToPoints.values():
      for ax, ay, _, _ in points:
        for cx, cy, dx, dy in points:
          # AC is perpendicular to AD.
          # AC dot AD = (cx - ax, cy - ay) dot (dx - ax, dy - ay) == 0.
          if (cx - ax) * (dx - ax) + (cy - ay) * (dy - ay) == 0:
            squaredArea = dist(ax, ay, cx, cy) * dist(ax, ay, dx, dy)
            if squaredArea > 0:
              ans = min(ans, squaredArea)

    return 0 if ans == math.inf else sqrt(ans)


# Link: https://leetcode.com/problems/kth-largest-element-in-an-array/description/
class Solution:
  def findKthLargest(self, nums: List[int], k: int) -> int:
    def quickSelect(l: int, r: int, k: int) -> int:
      randIndex = random.randint(0, r - l) + l
      nums[randIndex], nums[r] = nums[r], nums[randIndex]
      pivot = nums[r]

      nextSwapped = l
      for i in range(l, r):
        if nums[i] >= pivot:
          nums[nextSwapped], nums[i] = nums[i], nums[nextSwapped]
          nextSwapped += 1
      nums[nextSwapped], nums[r] = nums[r], nums[nextSwapped]

      count = nextSwapped - l + 1  # Number of nums >= pivot
      if count == k:
        return nums[nextSwapped]
      if count > k:
        return quickSelect(l, nextSwapped - 1, k)
      return quickSelect(nextSwapped + 1, r, k - count)

    return quickSelect(0, len(nums) - 1, k)


# Link: https://leetcode.com/problems/kth-largest-element-in-an-array/description/
class Solution:
  def findKthLargest(self, nums: List[int], k: int) -> int:
    def quickSelect(l: int, r: int, k: int) -> int:
      pivot = nums[r]

      nextSwapped = l
      for i in range(l, r):
        if nums[i] >= pivot:
          nums[nextSwapped], nums[i] = nums[i], nums[nextSwapped]
          nextSwapped += 1
      nums[nextSwapped], nums[r] = nums[r], nums[nextSwapped]

      count = nextSwapped - l + 1  # Number of nums >= pivot
      if count == k:
        return nums[nextSwapped]
      if count > k:
        return quickSelect(l, nextSwapped - 1, k)
      return quickSelect(nextSwapped + 1, r, k - count)

    return quickSelect(0, len(nums) - 1, k)


# Link: https://leetcode.com/problems/kth-largest-element-in-an-array/description/
class Solution:
  def findKthLargest(self, nums: List[int], k: int) -> int:
    minHeap = []

    for num in nums:
      heapq.heappush(minHeap, num)
      if len(minHeap) > k:
        heapq.heappop(minHeap)

    return minHeap[0]


# Link: https://leetcode.com/problems/count-nodes-with-the-highest-score/description/
class Solution:
  def countHighestScoreNodes(self, parents: List[int]) -> int:
    tree = [[] for _ in range(len(parents))]

    for i, parent in enumerate(parents):
      if parent == -1:
        continue
      tree[parent].append(i)

    ans = 0
    maxScore = 0

    def dfs(u: int) -> int:  # Returns node count
      nonlocal ans
      nonlocal maxScore
      count = 1
      score = 1
      for v in tree[u]:
        childCount = dfs(v)
        count += childCount
        score *= childCount
      score *= len(parents) - count or 1
      if score > maxScore:
        maxScore = score
        ans = 1
      elif score == maxScore:
        ans += 1
      return count

    dfs(0)
    return ans


# Link: https://leetcode.com/problems/k-radius-subarray-averages/description/
class Solution:
  def getAverages(self, nums: List[int], k: int) -> List[int]:
    n = len(nums)
    size = 2 * k + 1
    ans = [-1] * n
    if size > n:
      return ans

    summ = sum(nums[:size])

    for i in range(k, n - k):
      ans[i] = summ // size
      if i + k + 1 < n:
        summ += nums[i + k + 1] - nums[i - k]

    return ans


# Link: https://leetcode.com/problems/find-maximal-uncovered-ranges/description/
class Solution:
  def findMaximalUncoveredRanges(self, n: int, ranges: List[List[int]]) -> List[List[int]]:
    ans = []
    start = 0

    for l, r in sorted(ranges):
      if start < l:
        ans.append([start, l - 1])
      if start <= r:
        start = r + 1

    if start < n:
      ans.append([start, n - 1])

    return ans


# Link: https://leetcode.com/problems/unique-binary-search-trees/description/
class Solution:
  def numTrees(self, n: int) -> int:
    # dp[i] := the number of unique BST's that store values 1..i
    dp = [1, 1] + [0] * (n - 1)

    for i in range(2, n + 1):
      for j in range(i):
        dp[i] += dp[j] * dp[i - j - 1]

    return dp[n]


# Link: https://leetcode.com/problems/count-the-number-of-k-free-subsets/description/
class Solution:
  def countTheNumOfKFreeSubsets(self, nums: List[int], k: int) -> int:
    modToSubset = collections.defaultdict(set)

    for num in nums:
      modToSubset[num % k].add(num)

    prevNum = -k
    skip = 0
    pick = 0

    for subset in modToSubset.values():
      for num in sorted(subset):
        skip, pick = skip + pick, \
            1 + skip + (0 if num - prevNum == k else pick)
        prevNum = num

    return 1 + skip + pick


# Link: https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/description/
class Solution:
  def minSwaps(self, data: List[int]) -> int:
    k = data.count(1)
    ones = 0  # the number of ones in the window
    maxOnes = 0  # the maximum number of ones in the window

    for i, num in enumerate(data):
      if i >= k and data[i - k]:
        ones -= 1
      if num:
        ones += 1
      maxOnes = max(maxOnes, ones)

    return k - maxOnes


# Link: https://leetcode.com/problems/design-linked-list/description/
class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None


class MyLinkedList:
  def __init__(self):
    self.length = 0
    self.dummy = ListNode(0)

  def get(self, index: int) -> int:
    if index < 0 or index >= self.length:
      return -1
    curr = self.dummy.next
    for _ in range(index):
      curr = curr.next
    return curr.val

  def addAtHead(self, val: int) -> None:
    curr = self.dummy.next
    self.dummy.next = ListNode(val)
    self.dummy.next.next = curr
    self.length += 1

  def addAtTail(self, val: int) -> None:
    curr = self.dummy
    while curr.next:
      curr = curr.next
    curr.next = ListNode(val)
    self.length += 1

  def addAtIndex(self, index: int, val: int) -> None:
    if index > self.length:
      return
    curr = self.dummy
    for _ in range(index):
      curr = curr.next
    temp = curr.next
    curr.next = ListNode(val)
    curr.next.next = temp
    self.length += 1

  def deleteAtIndex(self, index: int) -> None:
    if index < 0 or index >= self.length:
      return
    curr = self.dummy
    for _ in range(index):
      curr = curr.next
    temp = curr.next
    curr.next = temp.next
    self.length -= 1


# Link: https://leetcode.com/problems/find-all-duplicates-in-an-array/description/
class Solution:
  def findDuplicates(self, nums: List[int]) -> List[int]:
    ans = []

    for num in nums:
      nums[abs(num) - 1] *= -1
      if nums[abs(num) - 1] > 0:
        ans.append(abs(num))

    return ans


# Link: https://leetcode.com/problems/gray-code/description/
class Solution:
  def grayCode(self, n: int) -> List[int]:
    ans = [0]

    for i in range(n):
      for j in reversed(range(len(ans))):
        ans.append(ans[j] | 1 << i)

    return ans


# Link: https://leetcode.com/problems/maximum-star-sum-of-a-graph/description/
class Solution:
  def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
    n = len(vals)
    ans = -math.inf
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append((v, vals[v]))
      graph[v].append((u, vals[u]))

    for i, starSum in enumerate(vals):
      maxHeap = []
      for _, val in graph[i]:
        if val > 0:
          heapq.heappush(maxHeap, -val)
      j = 0
      while j < k and maxHeap:
        starSum -= heapq.heappop(maxHeap)
        j += 1
      ans = max(ans, starSum)

    return ans


# Link: https://leetcode.com/problems/get-watched-videos-by-your-friends/description/
class Solution:
  def watchedVideosByFriends(self, watchedVideos: List[List[str]], friends: List[List[int]],
                             id: int, level: int) -> List[str]:
    visited = [False] * 100
    visited[id] = True
    q = collections.deque([id])
    count = collections.Counter()

    for _ in range(level):
      for _ in range(len(q)):
        curr = q.popleft()
        for friend in friends[curr]:
          if not visited[friend]:
            visited[friend] = True
            q.append(friend)

    for friend in q:
      for video in watchedVideos[friend]:
        count[video] += 1

    return sorted(count.keys(), key=lambda video: (count[video], video))


# Link: https://leetcode.com/problems/stepping-numbers/description/
class Solution:
  def countSteppingNumbers(self, low: int, high: int) -> List[int]:
    ans = [0] if low == 0 else []
    q = collections.deque(list(range(1, 10)))

    while q:
      curr = q.popleft()
      if curr > high:
        continue
      if curr >= low:
        ans.append(curr)
      lastDigit = curr % 10
      if lastDigit > 0:
        q.append(curr * 10 + lastDigit - 1)
      if lastDigit < 9:
        q.append(curr * 10 + lastDigit + 1)

    return ans


# Link: https://leetcode.com/problems/stepping-numbers/description/
class Solution:
  def countSteppingNumbers(self, low: int, high: int) -> List[int]:
    ans = [0] if low == 0 else []

    def dfs(curr: int) -> None:
      if curr > high:
        return
      if curr >= low:
        ans.append(curr)

      lastDigit = curr % 10
      if lastDigit > 0:
        dfs(curr * 10 + lastDigit - 1)
      if lastDigit < 9:
        dfs(curr * 10 + lastDigit + 1)

    for i in range(1, 9 + 1):
      dfs(i)

    ans.sort()
    return ans


# Link: https://leetcode.com/problems/minimum-moves-to-spread-stones-over-grid/description/
class Solution:
  def minimumMoves(self, grid: List[List[int]]) -> int:
    if sum(row.count(0) for row in grid) == 0:
      return 0

    ans = math.inf

    for i in range(3):
      for j in range(3):
        if grid[i][j] == 0:
          for x in range(3):
            for y in range(3):
              if grid[x][y] > 1:
                grid[x][y] -= 1
                grid[i][j] += 1
                ans = min(ans, abs(x - i) + abs(y - j) + self.minimumMoves(grid))
                grid[x][y] += 1
                grid[i][j] -= 1

    return ans


# Link: https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/description/
class Solution:
  def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
    m = len(mat)
    n = len(mat[0])
    ans = 0
    prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        prefix[i + 1][j + 1] = mat[i][j] + prefix[i][j + 1] + \
            prefix[i + 1][j] - prefix[i][j]

    def squareSum(r1: int, c1: int, r2: int, c2: int) -> int:
      return prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1] - prefix[r2 + 1][c1] + prefix[r1][c1]

    for i in range(m):
      for j in range(n):
        for length in range(ans, min(m - i, n - j)):
          if squareSum(i, j, i + length, j + length) > threshold:
            break
          ans = max(ans, length + 1)

    return ans


# Link: https://leetcode.com/problems/minimum-score-triangulation-of-polygon/description/
class Solution:
  def minScoreTriangulation(self, values: List[int]) -> int:
    n = len(values)
    dp = [[0] * n for _ in range(n)]

    for j in range(2, n):
      for i in range(j - 2, -1, -1):
        dp[i][j] = math.inf
        for k in range(i + 1, j):
          dp[i][j] = min(dp[i][j],
                         dp[i][k] + values[i] * values[k] * values[j] + dp[k][j])

    return dp[0][n - 1]


# Link: https://leetcode.com/problems/sentence-similarity-ii/description/
class Solution:
  def areSentencesSimilarTwo(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:
    if len(words1) != len(words2):
      return False

    # graph[key] := all the similar words of key
    graph = collections.defaultdict(set)

    for a, b in pairs:
      graph[a].add(b)
      graph[b].add(a)

    def dfs(word1: str, word2: str, seen: set) -> bool:
      if word1 in graph[word2]:
        return True

      seen.add(word1)

      for child in graph[word1]:
        if child in seen:
          continue
        if dfs(child, word2, seen):
          return True

      return False

    for word1, word2 in zip(words1, words2):
      if word1 == word2:
        continue
      if word1 not in graph:
        return False
      if not dfs(word1, word2, set()):
        return False

    return True


# Link: https://leetcode.com/problems/find-nearest-right-node-in-binary-tree/description/
class Solution:
  def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> Optional[TreeNode]:
    ans = None
    targetDepth = -1

    def dfs(root: TreeNode, depth: int) -> None:
      nonlocal ans
      nonlocal targetDepth
      if not root:
        return
      if root == u:
        targetDepth = depth
        return
      if depth == targetDepth and not ans:
        ans = root
        return
      dfs(root.left, depth + 1)
      dfs(root.right, depth + 1)

    dfs(root, 0)
    return ans


# Link: https://leetcode.com/problems/count-pairs-of-equal-substrings-with-minimum-difference/description/
class Solution:
  def countQuadruples(self, s1: str, s2: str) -> int:
    # To minimize j - a, the length of the substring should be 1. This is
    # because for substrings with a size greater than 1, a will decrease,
    # causing j - a to become larger.
    ans = 0
    diff = math.inf  # diff := j - a
    firstJ = {}
    lastA = {}

    for j in range(len(s1) - 1, -1, -1):
      firstJ[s1[j]] = j

    for a in range(len(s2)):
      lastA[s2[a]] = a

    for c in string.ascii_lowercase:
      if c not in firstJ or c not in lastA:
        continue
      if firstJ[c] - lastA[c] < diff:
        diff = firstJ[c] - lastA[c]
        ans = 0
      if firstJ[c] - lastA[c] == diff:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/bulb-switcher/description/
class Solution:
  def bulbSwitch(self, n: int) -> int:
    # The k-th bulb can only be switched when k % i == 0.
    # So, we can rephrase the problem:
    # To find number of numbers <= n that have odd factors.
    # Obviously, only square numbers have odd factor(s).
    # e.g. n = 10, only 1, 4, and 9 are square numbers that <= 10
    return int(sqrt(n))


# Link: https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/description/
class Solution:
  def maxAbsoluteSum(self, nums):
    summ = 0
    maxPrefix = 0
    minPrefix = 0

    for num in nums:
      summ += num
      maxPrefix = max(maxPrefix, summ)
      minPrefix = min(minPrefix, summ)

    return maxPrefix - minPrefix


# Link: https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/description/
class Solution:
  def maxAbsoluteSum(self, nums):
    ans = -math.inf
    maxSum = 0
    minSum = 0

    for num in nums:
      maxSum = max(num, maxSum + num)
      minSum = min(num, minSum + num)
      ans = max(ans, maxSum, -minSum)

    return ans


# Link: https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/description/
class FindElements:
  def __init__(self, root: Optional[TreeNode]):
    self.vals = set()
    self.dfs(root, 0)

  def find(self, target: int) -> bool:
    return target in self.vals

  def dfs(self, root: Optional[TreeNode], val: int) -> None:
    if not root:
      return

    root.val = val
    self.vals.add(val)
    self.dfs(root.left, val * 2 + 1)
    self.dfs(root.right, val * 2 + 2)


# Link: https://leetcode.com/problems/number-of-subarrays-with-lcm-equal-to-k/description/
class Solution:
  def subarrayLCM(self, nums: List[int], k: int) -> int:
    ans = 0

    for i, runningLcm in enumerate(nums):
      for j in range(i, len(nums)):
        runningLcm = math.lcm(runningLcm, nums[j])
        if runningLcm > k:
          break
        if runningLcm == k:
          ans += 1

    return ans


# Link: https://leetcode.com/problems/apply-bitwise-operations-to-make-strings-equal/description/
class Solution:
  def makeStringsEqual(self, s: str, target: str) -> bool:
    return ('1' in s) == ('1' in target)


# Link: https://leetcode.com/problems/minimum-cost-to-make-all-characters-equal/description/
class Solution:
  def minimumCost(self, s: str) -> int:
    n = len(s)
    ans = 0

    for i in range(1, n):
      if s[i] != s[i - 1]:
        # Invert s[0..i - 1] or s[i..n - 1].
        ans += min(i, n - i)

    return ans


# Link: https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/description/
class Solution:
  def numPairsDivisibleBy60(self, time: List[int]) -> int:
    ans = 0
    count = [0] * 60

    for t in time:
      t %= 60
      ans += count[(60 - t) % 60]
      count[t] += 1

    return ans


# Link: https://leetcode.com/problems/html-entity-parser/description/
class Solution:
  def entityParser(self, text: str) -> str:
    entityToChar = {'&quot;': '"', '&apos;': '\'',
                    '&gt;': '>', '&lt;': '<', '&frasl;': '/'}

    for entity, c in entityToChar.items():
      text = text.replace(entity, c)

    # Process '&' in last.
    return text.replace('&amp;', '&')


# Link: https://leetcode.com/problems/house-robber-iv/description/
class Solution:
  def minCapability(self, nums: List[int], k: int) -> int:
    def numStolenHouses(capacity: int) -> int:
      stolenHouses = 0
      i = 0
      while i < len(nums):
        if nums[i] <= capacity:
          stolenHouses += 1
          i += 1
        i += 1
      return stolenHouses

    return bisect.bisect_left(range(max(nums)), k,
                              key=lambda m: numStolenHouses(m))


# Link: https://leetcode.com/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/description/
class Solution:
  def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
    # leftStep + rightStep = k
    # rightStep - leftStep = endPos - startPos
    #        2 * rightStep = k + endPos - startPos
    #            rightStep = (k + endPos - startPos) // 2
    val = k + endPos - startPos
    if val < 0 or val & 1:
      return 0
    rightStep = val // 2
    leftStep = k - rightStep
    if leftStep < 0:
      return 0
    return self._nCk(leftStep + rightStep, min(leftStep, rightStep))

  # C(n, k) = C(n - 1, k) + C(n - 1, k - 1)
  def _nCk(self, n: int, k: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := C(n so far, i)
    dp = [1] + [0] * k

    for _ in range(n):  # Calculate n times.
      for j in range(k, 0, -1):
        dp[j] += dp[j - 1]
        dp[j] %= kMod

    return dp[k]


# Link: https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients/description/
class Solution:
  def numOfBurgers(self, tomatoSlices: int, cheeseSlices: int) -> List[int]:
    if tomatoSlices % 2 == 1 or tomatoSlices < 2 * cheeseSlices or tomatoSlices > cheeseSlices * 4:
      return []

    jumboBurgers = (tomatoSlices - 2 * cheeseSlices) // 2

    return [jumboBurgers, cheeseSlices - jumboBurgers]


# Link: https://leetcode.com/problems/continuous-subarray-sum/description/
class Solution:
  def checkSubarraySum(self, nums: List[int], k: int) -> bool:
    prefix = 0
    prefixToIndex = {0: -1}

    for i, num in enumerate(nums):
      prefix += num
      if k != 0:
        prefix %= k
      if prefix in prefixToIndex:
        if i - prefixToIndex[prefix] > 1:
          return True
      else:
        # Set a new key if it's absent because the previous index is better.
        prefixToIndex[prefix] = i

    return False


# Link: https://leetcode.com/problems/number-of-equal-count-substrings/description/
class Solution:
  def equalCountSubstrings(self, s: str, count: int) -> int:
    maxUnique = len(set(s))
    ans = 0

    for unique in range(1, maxUnique + 1):
      windowSize = unique * count
      lettersCount = collections.Counter()
      uniqueCount = 0
      for i, c in enumerate(s):
        lettersCount[c] += 1
        if lettersCount[c] == count:
          uniqueCount += 1
        if i >= windowSize:
          lettersCount[s[i - windowSize]] -= 1
          if lettersCount[s[i - windowSize]] == count - 1:
            uniqueCount -= 1
        ans += uniqueCount == unique

    return ans


# Link: https://leetcode.com/problems/lonely-pixel-ii/description/
class Solution:
  def findBlackPixel(self, picture: List[List[str]], target: int) -> int:
    m = len(picture)
    n = len(picture[0])
    ans = 0
    rows = [row.count('B') for row in picture]
    cols = [col.count('B') for col in zip(*picture)]
    rowStrings = [''.join(row) for row in picture]
    countRowStrings = collections.Counter(rowStrings)

    for i, (row, stringRow) in enumerate(zip(rows, rowStrings)):
      if row == target and countRowStrings[stringRow] == target:
        for j, col in enumerate(cols):
          if picture[i][j] == 'B' and col == target:
            ans += 1

    return ans


# Link: https://leetcode.com/problems/maximize-the-confusion-of-an-exam/description/
class Solution:
  def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
    ans = 0
    maxCount = 0
    count = collections.Counter()

    l = 0
    for r, c in enumerate(answerKey):
      count[c == 'T'] += 1
      maxCount = max(maxCount, count[c == 'T'])
      while maxCount + k < r - l + 1:
        count[answerKey[l] == 'T'] -= 1
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/
class Solution:
  def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
    potions.sort()

    def firstIndexSuccess(spell: int):
      """Returns the first index i s.t. spell * potions[i] >= success."""
      l = 0
      r = len(potions)
      while l < r:
        m = (l + r) // 2
        if spell * potions[m] >= success:
          r = m
        else:
          l = m + 1
      return l

    return [len(potions) - firstIndexSuccess(spell) for spell in spells]


