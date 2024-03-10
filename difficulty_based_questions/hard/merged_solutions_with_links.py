# Link: https://leetcode.com/problems/maximum-sum-queries/description/
class Pair:
  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y

  def __iter__(self):
    yield self.x
    yield self.y


class IndexedQuery:
  def __init__(self, queryIndex: int, minX: int, minY: int):
    self.queryIndex = queryIndex
    self.minX = minX
    self.minY = minY

  def __iter__(self):
    yield self.queryIndex
    yield self.minX
    yield self.minY


class Solution:
  def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
    pairs = sorted([Pair(nums1[i], nums2[i])
                   for i in range(len(nums1))], key=lambda p: p.x, reverse=True)
    ans = [0] * len(queries)
    stack = []  # [(y, x + y)]

    pairsIndex = 0
    for queryIndex, minX, minY in sorted([IndexedQuery(i, query[0], query[1])
                                          for i, query in enumerate(queries)],
                                         key=lambda iq: -iq.minX):
      while pairsIndex < len(pairs) and pairs[pairsIndex].x >= minX:
        # x + y is a better candidate. Given that x is decreasing, the
        # condition "x + y >= stack[-1][1]" suggests that y is relatively
        # larger, thereby making it a better candidate.
        x, y = pairs[pairsIndex]
        while stack and x + y >= stack[-1][1]:
          stack.pop()
        if not stack or y > stack[-1][0]:
          stack.append((y, x + y))
        pairsIndex += 1
      j = self._firstGreaterEqual(stack, minY)
      ans[queryIndex] = -1 if j == len(stack) else stack[j][1]

    return ans

  def _firstGreaterEqual(self, A: List[Tuple[int, int]], target: int) -> int:
    l = 0
    r = len(A)
    while l < r:
      m = (l + r) // 2
      if A[m][0] >= target:
        r = m
      else:
        l = m + 1
    return l


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/description/
class Solution:
  def minOperations(self, nums: List[int]) -> int:
    n = len(nums)
    ans = n
    nums = sorted(set(nums))

    for i, start in enumerate(nums):
      end = start + n - 1
      index = bisect_right(nums, end)
      uniqueLength = index - i
      ans = min(ans, n - uniqueLength)

    return ans


# Link: https://leetcode.com/problems/bus-routes/description/
class Solution:
  def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
    if source == target:
      return 0

    graph = collections.defaultdict(list)
    usedBuses = set()

    for i in range(len(routes)):
      for route in routes[i]:
        graph[route].append(i)

    ans = 0
    q = collections.deque([source])

    while q:
      ans += 1
      for _ in range(len(q)):
        for bus in graph[q.popleft()]:
          if bus in usedBuses:
            continue
          usedBuses.add(bus)
          for nextRoute in routes[bus]:
            if nextRoute == target:
              return ans
            q.append(nextRoute)

    return -1


# Link: https://leetcode.com/problems/count-prefix-and-suffix-pairs-ii/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[Tuple[str, str], TrieNode] = {}
    self.count = 0


class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, word: str) -> int:
    node = self.root
    count = 0
    for i, prefix in enumerate(word):
      suffix = word[-i - 1]
      node = node.children.setdefault((prefix, suffix), TrieNode())
      count += node.count
    node.count += 1
    return count


class Solution:
  # Same as 3045. Count Prefix and Suffix Pairs II
  def countPrefixSuffixPairs(self, words: List[str]) -> int:
    trie = Trie()
    return sum(trie.insert(word) for word in words)


# Link: https://leetcode.com/problems/redundant-connection-ii/description/
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
  def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
    ids = [0] * (len(edges) + 1)
    nodeWithTwoParents = 0

    for _, v in edges:
      ids[v] += 1
      if ids[v] == 2:
        nodeWithTwoParents = v

    def findRedundantDirectedConnection(skippedEdgeIndex: int) -> List[int]:
      uf = UnionFind(len(edges) + 1)

      for i, edge in enumerate(edges):
        if i == skippedEdgeIndex:
          continue
        if not uf.unionByRank(edge[0], edge[1]):
          return edge

      return []

    # If there is no edge with two ids, don't skip any edge.
    if nodeWithTwoParents == 0:
      return findRedundantDirectedConnection(-1)

    for i in reversed(range(len(edges))):
      _, v = edges[i]
      if v == nodeWithTwoParents:
        # Try to delete the edges[i].
        if not findRedundantDirectedConnection(i):
          return edges[i]


# Link: https://leetcode.com/problems/trapping-rain-water/description/
class Solution:
  def trap(self, height: List[int]) -> int:
    if not height:
      return 0

    ans = 0
    l = 0
    r = len(height) - 1
    maxL = height[l]
    maxR = height[r]

    while l < r:
      if maxL < maxR:
        ans += maxL - height[l]
        l += 1
        maxL = max(maxL, height[l])
      else:
        ans += maxR - height[r]
        r -= 1
        maxR = max(maxR, height[r])

    return ans


# Link: https://leetcode.com/problems/trapping-rain-water/description/
class Solution:
  def trap(self, height: List[int]) -> int:
    n = len(height)
    l = [0] * n  # l[i] := max(height[0..i])
    r = [0] * n  # r[i] := max(height[i..n))

    for i, h in enumerate(height):
      l[i] = h if i == 0 else max(h, l[i - 1])

    for i, h in reversed(list(enumerate(height))):
      r[i] = h if i == n - 1 else max(h, r[i + 1])

    return sum(min(l[i], r[i]) - h
               for i, h in enumerate(height))


# Link: https://leetcode.com/problems/best-meeting-point/description/
class Solution:
  def minTotalDistance(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    # i indices s.t. grid[i][j] == 1
    I = [i for i in range(m) for j in range(n) if grid[i][j]]
    # j indices s.t. grid[i][j] == 1
    J = [j for j in range(n) for i in range(m) if grid[i][j]]

    def minTotalDistance(grid: List[int]) -> int:
      summ = 0
      i = 0
      j = len(grid) - 1
      while i < j:
        summ += grid[j] - grid[i]
        i += 1
        j -= 1
      return summ

    # sum(i - median(I)) + sum(j - median(J))
    return minTotalDistance(I) + minTotalDistance(J)


# Link: https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities/description/
class Solution:
  def countSubgraphsForEachDiameter(self, n: int, edges: List[List[int]]) -> List[int]:
    maxMask = 1 << n
    dist = self._floydWarshall(n, edges)
    ans = [0] * (n - 1)

    # mask := the subset of the cities
    for mask in range(maxMask):
      maxDist = self._getMaxDist(mask, dist, n)
      if maxDist > 0:
        ans[maxDist - 1] += 1

    return ans

  def _floydWarshall(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    dist = [[n] * n for _ in range(n)]

    for i in range(n):
      dist[i][i] = 0

    for u, v in edges:
      dist[u - 1][v - 1] = 1
      dist[v - 1][u - 1] = 1

    for k in range(n):
      for i in range(n):
        for j in range(n):
          dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

  def _getMaxDist(self, mask: int, dist: List[List[int]], n: int) -> int:
    maxDist = 0
    edgeCount = 0
    cityCount = 0
    for u in range(n):
      if (mask >> u) & 1 == 0:  # u is not in the subset.
        continue
      cityCount += 1
      for v in range(u + 1, n):
        if (mask >> v) & 1 == 0:  # v is not in the subset.
          continue
        if dist[u][v] == 1:  # u and v are connected.
          edgeCount += 1
        maxDist = max(maxDist, dist[u][v])

    return maxDist if edgeCount == cityCount - 1 else 0


# Link: https://leetcode.com/problems/minimize-the-total-price-of-the-trips/description/
class Solution:
  def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int],
                        trips: List[List[int]]) -> int:
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    # count[i] := the number of times i is traversed
    count = [0] * n

    def dfsCount(u: int, prev: int, end: int, path: List[int]) -> None:
      path.append(u)
      if u == end:
        for i in path:
          count[i] += 1
        return
      for v in graph[u]:
        if v != prev:
          dfsCount(v, u, end,  path)
      path.pop()

    for start, end in trips:
      dfsCount(start, -1, end, [])

    @functools.lru_cache(None)
    def dfs(u: int, prev: int, parentHalved: bool) -> int:
      """
      Returns the minimum price sum for the i-th node, where its parent is
      halved parent or not halved not.
      """
      sumWithFullNode = price[u] * count[u] + sum(dfs(v, u, False)
                                                  for v in graph[u]
                                                  if v != prev)
      if parentHalved:  # Can't halve this node if its parent was halved.
        return sumWithFullNode
      sumWithHalvedNode = (price[u] // 2) * count[u] + sum(dfs(v, u, True)
                                                           for v in graph[u]
                                                           if v != prev)
      return min(sumWithFullNode, sumWithHalvedNode)

    return dfs(0, -1, False)


# Link: https://leetcode.com/problems/least-operators-to-express-number/description/
class Solution:
  def leastOpsExpressTarget(self, x: int, target: int) -> int:
    @functools.lru_cache(None)
    def dfs(target):
      if x > target:
        return min(2 * target - 1, 2 * (x - target))
      if x == target:
        return 0

      prod = x
      n = 0
      while prod < target:
        prod *= x
        n += 1
      if prod == target:
        return n

      ans = dfs(target - prod // x) + n
      if prod < 2 * target:
        ans = min(ans, dfs(prod - target) + n + 1)
      return ans

    return dfs(target)


# Link: https://leetcode.com/problems/number-of-distinct-islands-ii/description/
class Solution:
  def numDistinctIslands2(self, grid: List[List[int]]) -> int:
    seen = set()

    def dfs(i: int, j: int):
      if i < 0 or i == len(grid) or j < 0 or j == len(grid[0]):
        return
      if grid[i][j] == 0 or (i, j) in seen:
        return

      seen.add((i, j))
      island.append((i, j))
      dfs(i + 1, j)
      dfs(i - 1, j)
      dfs(i, j + 1)
      dfs(i, j - 1)

    def normalize(island: List[tuple]) -> List[tuple]:
      # points[i] := 8 different rotations/reflections of an island
      points = [[] for _ in range(8)]

      for i, j in island:
        points[0].append((i, j))
        points[1].append((i, -j))
        points[2].append((-i, j))
        points[3].append((-i, -j))
        points[4].append((j, i))
        points[5].append((j, -i))
        points[6].append((-j, i))
        points[7].append((-j, -i))

      points = [sorted(p) for p in points]

      # Normalize each p by substracting p[1..7] with p[0].
      for p in points:
        for i in range(1, len(island)):
          p[i] = (p[i][0] - p[0][0],
                  p[i][1] - p[0][1])
        p[0] = (0, 0)

      return sorted(points)[0]

    islands = set()  # all the islands with different shapes

    for i in range(len(grid)):
      for j in range(len(grid[0])):
        island = []
        dfs(i, j)
        if island:
          islands.add(frozenset(normalize(island)))

    return len(islands)


# Link: https://leetcode.com/problems/sum-of-imbalance-numbers-of-all-subarrays/description/
class Solution:
  # If sorted(nums)[i + 1] - sorted(nums)[i] > 1, then there's a gap. Instead
  # of determining the number of gaps in each subarray, let's find out how many
  # subarrays contain each gap.
  def sumImbalanceNumbers(self, nums: List[int]) -> int:
    n = len(nums)
    # Note that to avoid double counting, only `left` needs to check nums[i].
    # This adjustment ensures that i represents the position of the leftmost
    # element of nums[i] within the subarray.

    # left[i] := the maximum index l s.t. nums[l] = nums[i] or nums[i] + 1
    left = [0] * n
    # right[i] := the minimum index r s.t. nums[r] = nums[i]
    right = [0] * n

    numToIndex = [-1] * (n + 2)
    for i, num in enumerate(nums):
      left[i] = max(numToIndex[num], numToIndex[num + 1])
      numToIndex[num] = i

    numToIndex = [n] * (n + 2)
    for i in range(n - 1, -1, -1):
      right[i] = numToIndex[nums[i] + 1]
      numToIndex[nums[i]] = i

    # The gap above nums[i] persists until encountering nums[i] or nums[i] + 1.
    # Consider subarrays nums[l..r] with l <= i <= r, where l in [left[i], i]
    # and r in [i, right[i] - 1]. There are (i - left[i]) * (right[i] - i)
    # subarrays satisfying this condition.
    #
    # Subtract n * (n + 1) / 2 to account for the overcounting of elements
    # initially assumed to have a gap. This adjustment is necessary as the
    # maximum element of every subarray does not have a gap.
    return sum((i - left[i]) * (right[i] - i)
               for i in range(n)) - n * (n + 1) // 2


# Link: https://leetcode.com/problems/maximum-number-of-intersections-on-the-chart/description/
class Solution:
  def maxIntersectionCount(self, y: List[int]) -> int:
    ans = 0
    intersectionCount = 0
    line = collections.Counter()

    for i, (a, b) in enumerate(itertools.pairwise(y)):
      start = 2 * a
      end = 2 * b + (0 if i == len(y) - 2 else -1 if b > a else 1)
      line[min(start, end)] += 1
      line[max(start, end) + 1] -= 1

    for count in sorted(line):
      intersectionCount += line[count]
      ans = max(ans, intersectionCount)

    return ans


# Link: https://leetcode.com/problems/tree-of-coprimes/description/
class Solution:
  def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
    kMax = 50
    ans = [-1] * len(nums)
    tree = [[] for _ in range(len(nums))]
    # stacks[i] := (node, depth)s of nodes with value i
    stacks = [[] for _ in range(kMax + 1)]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def getAncestor(u: int) -> int:
      maxNode = -1
      maxDepth = -1
      for i, stack in enumerate(stacks):
        if stack and stack[-1][1] > maxDepth and math.gcd(nums[u], i) == 1:
          maxNode, maxDepth = stack[-1]
      return maxNode

    def dfs(u: int, prev: int, depth: int) -> int:
      ans[u] = getAncestor(u)
      stacks[nums[u]].append((u, depth))

      for v in tree[u]:
        if prev != v:
          dfs(v, u, depth + 1)

      stacks[nums[u]].pop()

    dfs(0, -1, 0)
    return ans


# Link: https://leetcode.com/problems/minimum-time-takes-to-reach-destination-without-drowning/description/
class Solution:
  def minimumSeconds(self, land: List[List[str]]) -> int:
    self.dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(land)
    n = len(land[0])
    floodDist = self._getFloodDist(land)
    startPos = self._getStartPos(land, 'S')

    q = collections.deque([startPos])
    seen = {startPos}

    step = 1
    while q:
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in self.dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if land[x][y] == 'D':
            return step
          if floodDist[x][y] <= step or land[x][y] == 'X' or (x, y) in seen:
            continue
          q.append((x, y))
          seen.add((x, y))
      step += 1

    return -1

  def _getFloodDist(self, land: List[List[str]]) -> List[List[int]]:
    m = len(land)
    n = len(land[0])
    dist = [[math.inf] * n for _ in range(m)]
    q = collections.deque()
    seen = set()

    for i, row in enumerate(land):
      for j, cell in enumerate(row):
        if cell == '*':
          q.append((i, j))
          seen.add((i, j))

    d = 0
    while q:
      for _ in range(len(q)):
        i, j = q.popleft()
        dist[i][j] = d
        for dx, dy in self.dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if land[x][y] in 'XD' or (x, y) in seen:
            continue
          q.append((x, y))
          seen.add((x, y))
      d += 1

    return dist

  def _getStartPos(self, land: List[List[str]], c: str) -> Tuple[int, int]:
    for i, row in enumerate(land):
      for j, cell in enumerate(row):
        if cell == c:
          return i, j


# Link: https://leetcode.com/problems/escape-the-spreading-fire/description/
class Solution:
  def maximumMinutes(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    kMax = len(grid) * len(grid[0])
    fireGrid = [[-1] * len(grid[0]) for _ in range(len(grid[0]))]
    self._buildFireGrid(grid, fireGrid, dirs)

    ans = -1
    l = 0
    r = kMax

    while l <= r:
      m = (l + r) // 2
      if self._canStayFor(grid, fireGrid, m, dirs):
        ans = m
        l = m + 1
      else:
        r = m - 1

    return 1e9 if ans == kMax else ans

  def _buildFireGrid(self, grid: List[List[int]], fireMinute: List[List[int]], dirs: List[int]) -> None:
    minuteFromFire = 0
    q = collections.deque()

    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if grid[i][j] == 1:  # the fire
          q.append((i, j))
          fireMinute[i][j] = 0

    while q:
      minuteFromFire += 1
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == len(grid) or y < 0 or y == len(grid[0]):
            continue
          if grid[x][y] == 2:  # the wall
            continue
          if fireMinute[x][y] != -1:
            continue
          fireMinute[x][y] = minuteFromFire
          q.append((x, y))

  def _canStayFor(self, grid: List[List[int]], fireMinute: List[List[int]], minute: int, dirs: List[int]) -> bool:
    q = collections.deque([(0, 0)])  # the start position
    seen = {(0, 0)}

    while q:
      minute += 1
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == len(grid) or y < 0 or y == len(grid[0]):
            continue
          if grid[x][y] == 2:  # the wall
            continue
          if x == len(grid) - 1 and y == len(grid[0]) - 1:
            if fireMinute[x][y] != -1 and fireMinute[x][y] < minute:
              continue
            return True
          if fireMinute[x][y] != -1 and fireMinute[x][y] <= minute:
            continue
          if seen[x][y]:
            continue
          q.append((x, y))
          seen.add((x, y))

    return False


# Link: https://leetcode.com/problems/sum-of-distances-in-tree/description/
class Solution:
  def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
    ans = [0] * n
    count = [1] * n
    tree = collections.defaultdict(set)

    for u, v in edges:
      tree[u].add(v)
      tree[v].add(u)

    def postorder(node, parent=None):
      for child in tree[node]:
        if child == parent:
          continue
        postorder(child, node)
        count[node] += count[child]
        ans[node] += ans[child] + count[child]

    def preorder(node, parent=None):
      for child in tree[node]:
        if child == parent:
          continue
        # count[child] nodes are 1 step closer from child than parent.
        # (n - count[child]) nodes are 1 step farther from child than parent.
        ans[child] = ans[node] - count[child] + (n - count[child])
        preorder(child, node)

    postorder(0)
    preorder(0)
    return ans


# Link: https://leetcode.com/problems/maximum-number-of-ones/description/
class Solution:
  def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
    submatrix = [[0] * sideLength for _ in range(sideLength)]

    for i in range(width):
      for j in range(height):
        submatrix[i % sideLength][j % sideLength] += 1

    return sum(heapq.nlargest(maxOnes, [a for row in submatrix for a in row]))


# Link: https://leetcode.com/problems/maximum-number-of-ones/description/
class Solution:
  def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
    subCount = []

    def getCount(length: int, index: int) -> int:
      return (length - index - 1) // sideLength + 1

    for i in range(sideLength):
      for j in range(sideLength):
        subCount.append(getCount(width, i) * getCount(height, j))

    return sum(sorted(subCount, reverse=True)[:maxOnes])


# Link: https://leetcode.com/problems/max-chunks-to-make-sorted-ii/description/
class Solution:
  def maxChunksToSorted(self, arr: List[int]) -> int:
    n = len(arr)
    ans = 0
    maxi = -math.inf
    mini = [arr[-1]] * n

    for i in reversed(range(n - 1)):
      mini[i] = min(mini[i + 1], arr[i])

    for i in range(n - 1):
      maxi = max(maxi, arr[i])
      if maxi <= mini[i + 1]:
        ans += 1

    return ans + 1


# Link: https://leetcode.com/problems/my-calendar-iii/description/
from sortedcontainers import SortedDict


class MyCalendarThree:
  def __init__(self):
    self.timeline = SortedDict()

  def book(self, start: int, end: int) -> int:
    self.timeline[start] = self.timeline.get(start, 0) + 1
    self.timeline[end] = self.timeline.get(end, 0) - 1

    ans = 0
    activeEvents = 0

    for count in self.timeline.values():
      activeEvents += count
      ans = max(ans, activeEvents)

    return ans


# Link: https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/
from sortedcontainers import SortedList


class Solution:
  def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
    m = len(matrix)
    n = len(matrix[0])
    ans = -math.inf

    for baseCol in range(n):
      # sums[i] := sum(matrix[i][baseCol..j])
      sums = [0] * m
      for j in range(baseCol, n):
        for i in range(m):
          sums[i] += matrix[i][j]
        # Find the maximum sum <= k of all the subarrays.
        accumulate = SortedList([0])
        prefix = 0
        for summ in sums:
          prefix += summ
          it = accumulate.bisect_left(prefix - k)
          if it != len(accumulate):
            ans = max(ans, prefix - accumulate[it])
          accumulate.add(prefix)

    return ans


# Link: https://leetcode.com/problems/three-equal-parts/description/
class Solution:
  def threeEqualParts(self, arr: List[int]) -> List[int]:
    ones = sum(a == 1 for a in arr)

    if ones == 0:
      return [0, len(arr) - 1]
    if ones % 3 != 0:
      return [-1, -1]

    k = ones // 3
    i = 0

    for i in range(len(arr)):
      if arr[i] == 1:
        first = i
        break

    gapOnes = k

    for j in range(i + 1, len(arr)):
      if arr[j] == 1:
        gapOnes -= 1
        if gapOnes == 0:
          second = j
          break

    gapOnes = k

    for i in range(j + 1, len(arr)):
      if arr[i] == 1:
        gapOnes -= 1
        if gapOnes == 0:
          third = i
          break

    while third < len(arr) and arr[first] == arr[second] == arr[third]:
      first += 1
      second += 1
      third += 1

    if third == len(arr):
      return [first - 1, second]
    return [-1, -1]


# Link: https://leetcode.com/problems/minimum-operations-to-make-the-array-k-increasing/description/
class Solution:
  def kIncreasing(self, arr: List[int], k: int) -> int:
    def numReplaced(A: List[int]) -> int:
      tail = []
      for a in A:
        if not tail or tail[-1] <= a:
          tail.append(a)
        else:
          tail[bisect_right(tail, a)] = a
      return len(A) - len(tail)

    return sum(numReplaced(arr[i::k]) for i in range(k))


# Link: https://leetcode.com/problems/expression-add-operators/description/
class Solution:
  def addOperators(self, num: str, target: int) -> List[str]:
    ans = []

    def dfs(start: int, prev: int, eval: int, path: List[str]) -> None:
      if start == len(num):
        if eval == target:
          ans.append(''.join(path))
        return

      for i in range(start, len(num)):
        if i > start and num[start] == '0':
          return
        s = num[start:i + 1]
        curr = int(s)
        if start == 0:
          path.append(s)
          dfs(i + 1, curr, curr, path)
          path.pop()
        else:
          for op in ['+', '-', '*']:
            path.append(op + s)
            if op == '+':
              dfs(i + 1, curr, eval + curr, path)
            elif op == '-':
              dfs(i + 1, -curr, eval - curr, path)
            else:
              dfs(i + 1, prev * curr, eval - prev + prev * curr, path)
            path.pop()

    dfs(0, 0, 0, [])
    return ans


# Link: https://leetcode.com/problems/minimum-time-to-make-array-sum-at-most-x/description/
class Solution:
  def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
    n = len(nums1)
    # dp[j] := the maximum reduced value if we do j operations on the numbers
    # so far
    dp = [0] * (n + 1)
    sum1 = sum(nums1)
    sum2 = sum(nums2)

    for i, (num2, num1) in enumerate(sorted(zip(nums2, nums1)), 1):
      for j in range(i, 0, -1):
        dp[j] = max(
            # the maximum reduced value if we do j operations on the first
            # i - 1 numbers
            dp[j],
            # the maximum reduced value if we do j - 1 operations on the first
            # i - 1 numbers + making the i-th number of `nums1` to 0 at the
            # j-th operation
            dp[j - 1] + num2 * j + num1
        )

    for op in range(n + 1):
      if sum1 + sum2 * op - dp[op] <= x:
        return op

    return -1


# Link: https://leetcode.com/problems/minimum-time-to-make-array-sum-at-most-x/description/
class Solution:
  def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
    n = len(nums1)
    # dp[i][j] := the maximum reduced value if we do j operations on the first
    # i numbers
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    sum1 = sum(nums1)
    sum2 = sum(nums2)

    for i, (num2, num1) in enumerate(sorted(zip(nums2, nums1)), 1):
      for j in range(1, i + 1):
        dp[i][j] = max(
            # the maximum reduced value if we do j operations on the first
            # i - 1 numbers
            dp[i - 1][j],
            # the maximum reduced value if we do j - 1 operations on the first
            # i - 1 numbers + making the i-th number of `nums1` to 0 at the
            # j-th operation
            dp[i - 1][j - 1] + num2 * j + num1
        )

    for op in range(n + 1):
      if sum1 + sum2 * op - dp[n][op] <= x:
        return op

    return -1


# Link: https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/
class Solution:
  def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
    m = len(matrix)
    n = len(matrix[0])
    ans = 0

    # Transfer each row in the matrix to the prefix sum.
    for row in matrix:
      for i in range(1, n):
        row[i] += row[i - 1]

    for baseCol in range(n):
      for j in range(baseCol, n):
        prefixCount = collections.Counter({0: 1})
        summ = 0
        for i in range(m):
          if baseCol > 0:
            summ -= matrix[i][baseCol - 1]
          summ += matrix[i][j]
          ans += prefixCount[summ - target]
          prefixCount[summ] += 1

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/description/
class Solution:
  def minCost(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
    dp = [[-1] * n for _ in range(m)]
    q = collections.deque()

    def dfs(i: int, j: int, cost: int) -> None:
      if i < 0 or i == m or j < 0 or j == n:
        return
      if dp[i][j] != -1:
        return

      dp[i][j] = cost
      q.append((i, j))
      dx, dy = dirs[grid[i][j] - 1]
      dfs(i + dx, j + dy, cost)

    dfs(0, 0, 0)

    cost = 0
    while q:
      cost += 1
      for _ in range(len(q)):
        i, j = q.popleft()
        for dx, dy in dirs:
          dfs(i + dx, j + dy, cost)

    return dp[-1][-1]


# Link: https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/description/
class Solution:
  def minimumMountainRemovals(self, nums: List[int]) -> int:
    left = self._lengthOfLIS(nums)
    right = self._lengthOfLIS(nums[::-1])[::-1]
    maxMountainSeq = 0

    for l, r in zip(left, right):
      if l > 1 and r > 1:
        maxMountainSeq = max(maxMountainSeq, l + r - 1)

    return len(nums) - maxMountainSeq

  def _lengthOfLIS(self, nums: List[int]) -> List[int]:
    # tail[i] := the minimum tail of all the increasing subsequences having
    # length i + 1
    # It's easy to see that tail must be an increasing array.
    tail = []
    # dp[i] := the length of LIS ending in nums[i]
    dp = []

    for num in nums:
      if not tail or num > tail[-1]:
        tail.append(num)
      else:
        tail[bisect.bisect_left(tail, num)] = num
      dp.append(len(tail))

    return dp


# Link: https://leetcode.com/problems/largest-component-size-by-common-factor/description/
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
  def largestComponentSize(self, nums: List[int]) -> int:
    ans = 0
    uf = UnionFind(max(nums) + 1)
    count = collections.Counter()

    for num in nums:
      for x in range(2, int(math.sqrt(num) + 1)):
        if num % x == 0:
          uf.unionByRank(num, x)
          uf.unionByRank(num, num // x)

    for num in nums:
      numRoot = uf.find(num)
      count[numRoot] += 1
      ans = max(ans, count[numRoot])

    return ans


# Link: https://leetcode.com/problems/design-search-autocomplete-system/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.s: Optional[str] = None
    self.time = 0
    self.top3: List[TrieNode] = []

  def __lt__(self, other):
    if self.time == other.time:
      return self.s < other.s
    return self.time > other.time

  def update(self, node) -> None:
    if node not in self.top3:
      self.top3.append(node)
    self.top3.sort()
    if len(self.top3) > 3:
      self.top3.pop()


class AutocompleteSystem:
  def __init__(self, sentences: List[str], times: List[int]):
    self.root = TrieNode()
    self.curr = self.root
    self.s: List[chr] = []

    for sentence, time in zip(sentences, times):
      self._insert(sentence, time)

  def input(self, c: str) -> List[str]:
    if c == '#':
      self._insert(''.join(self.s), 1)
      self.curr = self.root
      self.s = []
      return []

    self.s.append(c)

    if self.curr:
      self.curr = self.curr.children.get(c, None)
    if not self.curr:
      return []
    return [node.s for node in self.curr.top3]

  def _insert(self, sentence: str, time: int) -> None:
    node = self.root
    for c in sentence:
      node = node.children.setdefault(c, TrieNode())
    node.s = sentence
    node.time += time

    leaf = node
    node: TrieNode = self.root
    for c in sentence:
      node = node.children[c]
      node.update(leaf)


# Link: https://leetcode.com/problems/frog-position-after-t-seconds/description/
class Solution:
  def frogPosition(self, n: int, edges: List[List[int]], t: int, target: int) -> float:
    tree = [[] for _ in range(n + 1)]
    q = collections.deque([1])
    seen = [False] * (n + 1)
    prob = [0] * (n + 1)

    prob[1] = 1
    seen[1] = True

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    for _ in range(t):
      for _ in range(len(q)):
        a = q.popleft()
        nChildren = sum(not seen[b] for b in tree[a])
        for b in tree[a]:
          if seen[b]:
            continue
          seen[b] = True
          prob[b] = prob[a] / nChildren
          q.append(b)
        if nChildren > 0:
          prob[a] = 0

    return prob[target]


# Link: https://leetcode.com/problems/count-good-triplets-in-an-array/description/
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
  def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
    n = len(nums1)
    numToIndex = {num: i for i, num in enumerate(nums1)}
    # Remap each number in `nums2` to the according index in `nums1` as `A`.
    # Rephrase the problem as finding the number of increasing tripets in `A`.
    A = [numToIndex[num] for num in nums2]
    # leftSmaller[i] := the number of A[j] < A[i], where 0 <= j < i
    leftSmaller = [0] * n
    # rightLarger[i] := the number of A[j] > A[i], where i < j < n
    rightLarger = [0] * n
    tree1 = FenwickTree(n)  # Calculates `leftSmaller`.
    tree2 = FenwickTree(n)  # Calculates `rightLarger`.

    for i, a in enumerate(A):
      leftSmaller[i] = tree1.get(a)
      tree1.update(a + 1, 1)

    for i, a in reversed(list(enumerate(A))):
      rightLarger[i] = tree2.get(n) - tree2.get(a)
      tree2.update(a + 1, 1)

    return sum(a * b for a, b in zip(leftSmaller, rightLarger))


# Link: https://leetcode.com/problems/count-the-number-of-k-big-indices/description/
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
  def kBigIndices(self, nums: List[int], k: int) -> int:
    n = len(nums)
    leftTree = FenwickTree(n)
    rightTree = FenwickTree(n)
    # left[i] := the number of `nums` < nums[i] with index < i
    left = [0] * n
    # right[i] := the number of `nums` < nums[i] with index > i
    right = [0] * n

    for i, num in enumerate(nums):
      left[i] = leftTree.get(num - 1)
      leftTree.update(num, 1)

    for i in range(n - 1, -1, -1):
      right[i] = rightTree.get(nums[i] - 1)
      rightTree.update(nums[i], 1)

    return sum(l >= k and r >= k for l, r in zip(left, right))


# Link: https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/description/
class Solution:
  def minNumberOperations(self, target: List[int]) -> int:
    ans = target[0]

    for a, b in zip(target, target[1:]):
      if a < b:
        ans += b - a

    return ans


# Link: https://leetcode.com/problems/patching-array/description/
class Solution:
  def minPatches(self, nums: List[int], n: int) -> int:
    ans = 0
    i = 0  # nums' index
    miss = 1  # the minimum sum in [1, n] we might miss

    while miss <= n:
      if i < len(nums) and nums[i] <= miss:
        miss += nums[i]
        i += 1
      else:
        # Greedily add `miss` itself to increase the range from
        # [1, miss) to [1, 2 * miss).
        miss += miss
        ans += 1

    return ans


# Link: https://leetcode.com/problems/count-fertile-pyramids-in-a-land/description/
class Solution:
  def countPyramids(self, grid: List[List[int]]) -> int:
    # dp[i][j] := the maximum height of the pyramid for which it is the apex
    def count(dp: List[List[int]]) -> int:
      ans = 0
      for i in range(len(dp) - 2, -1, -1):
        for j in range(1, len(dp[0]) - 1):
          if dp[i][j] == 1:
            dp[i][j] = min(dp[i + 1][j - 1],
                           dp[i + 1][j],
                           dp[i + 1][j + 1]) + 1
            ans += dp[i][j] - 1
      return ans

    return count(deepcopy(grid)[::-1]) + count(grid)


# Link: https://leetcode.com/problems/maximum-average-subarray-ii/description/
class Solution:
  def findMaxAverage(self, nums: List[int], k: int) -> float:
    kErr = 1e-5
    l = min(nums)
    r = max(nums)

    def check(m: float) -> bool:
      """
      Returns True if there's a subarray, where its length >= k and its average
      sum >= m.
      """
      summ = 0
      prevSum = 0
      minPrevSum = 0

      for i, num in enumerate(nums):
        # Need to substract m for each `num` so that we can check if the sum of
        # the subarray >= 0.
        summ += num - m
        if i >= k:
          prevSum += nums[i - k] - m
          minPrevSum = min(minPrevSum, prevSum)
        if i + 1 >= k and summ >= minPrevSum:
          return True

      return False

    while r - l > kErr:
      m = (l + r) / 2
      if check(m):
        l = m
      else:
        r = m

    return l


# Link: https://leetcode.com/problems/remove-invalid-parentheses/description/
class Solution:
  def removeInvalidParentheses(self, s: str) -> List[str]:
    # Similar to 921. Minimum Add to Make Parentheses Valid
    def getLeftAndRightCounts(s: str) -> Tuple[int, int]:
      """Returns how many '(' and ')' need to be deleted."""
      l = 0
      r = 0

      for c in s:
        if c == '(':
          l += 1
        elif c == ')':
          if l == 0:
            r += 1
          else:
            l -= 1

      return l, r

    def isValid(s: str):
      opened = 0  # the number of '(' - # of ')'
      for c in s:
        if c == '(':
          opened += 1
        elif c == ')':
          opened -= 1
        if opened < 0:
          return False
      return True  # opened == 0

    ans = []

    def dfs(s: str, start: int, l: int, r: int) -> None:
      if l == 0 and r == 0 and isValid(s):
        ans.append(s)
        return

      for i in range(start, len(s)):
        if i > start and s[i] == s[i - 1]:
          continue
        if r > 0 and s[i] == ')':  # Delete s[i]
          dfs(s[:i] + s[i + 1:], i, l, r - 1)
        elif l > 0 and s[i] == '(':  # Delete s[i]
          dfs(s[:i] + s[i + 1:], i, l - 1, r)

    l, r = getLeftAndRightCounts(s)
    dfs(s, 0, l, r)
    return ans


# Link: https://leetcode.com/problems/minimum-unique-word-abbreviation/description/
class Solution:
  def minAbbreviation(self, target: str, dictionary: List[str]) -> str:
    m = len(target)

    def getMask(word: str) -> int:
      # mask[i] = 0 := target[i] == word[i]
      # mask[i] = 1 := target[i] != word[i]
      # e.g. target = "apple"
      #        word = "blade"
      #        mask =  11110
      mask = 0
      for i, c in enumerate(word):
        if c != target[i]:
          mask |= 1 << m - 1 - i
      return mask

    masks = [getMask(word) for word in dictionary if len(word) == m]
    if not masks:
      return str(m)

    abbrs = []

    def getAbbr(cand: int) -> str:
      abbr = []
      replacedCount = 0
      for i, c in enumerate(target):
        if cand >> m - 1 - i & 1:
          # If cand[i] = 1, `abbr` should show the original character.
          if replacedCount:
            abbr += str(replacedCount)
          abbr.append(c)
          replacedCount = 0
        else:
          # If cand[i] = 0, `abbr` can be replaced.
          replacedCount += 1
      if replacedCount:
        abbr.append(str(replacedCount))
      return ''.join(abbr)

    # all the candidate representation of the target
    for cand in range(2**m):
      # All the masks have at lease one bit different from the candidate.
      if all(cand & mask for mask in masks):
        abbr = getAbbr(cand)
        abbrs.append(abbr)

    def getAbbrLen(abbr: str) -> int:
      abbrLen = 0
      i = 0
      j = 0
      while i < len(abbr):
        if abbr[j].isalpha():
          j += 1
        else:
          while j < len(abbr) and abbr[j].isdigit():
            j += 1
        abbrLen += 1
        i = j
      return abbrLen

    return min(abbrs, key=lambda x: getAbbrLen(x))


# Link: https://leetcode.com/problems/sequentially-ordinal-rank-tracker/description/
class Location:
  def __init__(self, name: str, score: int):
    self.name = name
    self.score = score

  def __lt__(self, location):
    if self.score == location.score:
      return self.name > location.name
    return self.score < location.score


class SORTracker:
  def __init__(self):
    self.l = []
    self.r = []
    self.k = 0  the number of times get() called

  def add(self, name: str, score: int) -> None:
    heapq.heappush(self.l, Location(name, score))
    if len(self.l) > self.k + 1:
      location = heapq.heappop(self.l)
      heapq.heappush(self.r, (-location.score, location.name))

  def get(self) -> str:
    name = self.l[0].name
    if self.r:
      topScore, topName = heapq.heappop(self.r)
      heapq.heappush(self.l, Location(topName, -topScore))
    self.k += 1
    return name


# Link: https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/description/
class Solution:
  def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
    ans = []
    binaryCount = collections.Counter()

    for word in words:
      mask = 0
      for c in word:
        mask |= 1 << (ord(c) - ord('a'))
      binaryCount[mask] += 1

    for puzzle in puzzles:
      valid = 0
      n = len(puzzle) - 1
      for i in range(1 << n):
        mask = 1 << ord(puzzle[0]) - ord('a')
        for j in range(n):
          if i & 1 << j:
            mask |= 1 << ord(puzzle[j + 1]) - ord('a')
        if mask in binaryCount:
          valid += binaryCount[mask]
      ans.append(valid)

    return ans


# Link: https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/
class Solution:
  def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
    ans = []
    xToNodes = collections.defaultdict(list)

    def dfs(node: Optional[TreeNode], x: int, y: int) -> None:
      if not node:
        return
      xToNodes[x].append((-y, node.val))
      dfs(node.left, x - 1, y - 1)
      dfs(node.right, x + 1, y - 1)

    dfs(root, 0, 0)

    for _, nodes in sorted(xToNodes.items(), key=lambda item: item[0]):
      ans.append([val for _, val in sorted(nodes)])

    return ans


# Link: https://leetcode.com/problems/minimum-relative-loss-after-buying-chocolates/description/
class Solution:
  def minimumRelativeLosses(self, prices: List[int], queries: List[List[int]]) -> List[int]:
    ans = []

    prices.sort()

    prefix = [0] + list(itertools.accumulate(prices))

    for k, m in queries:
      countFront = self._getCountFront(k, m, prices, prefix)
      countBack = m - countFront
      ans.append(self._getRelativeLoss(countFront, countBack, k, prefix))

    return ans

  def _getCountFront(self, k: int, m: int, prices: List[int], prefix: List[int]) -> int:
    """Returns `countFront` for query (k, m).


    Returns `countFront` for query (k, m) s.t. picking the first `countFront`
    and the last `m - countFront` chocolates is optimal.

    Define loss[i] := the relative loss of picking `prices[i]`.
    1. For prices[i] <= k, Bob pays prices[i] while Alice pays 0.
       Thus, loss[i] = prices[i] - 0 = prices[i].
    2. For prices[i] > k, Bob pays k while Alice pays prices[i] - k.
       Thus, loss[i] = k - (prices[i] - k) = 2 * k - prices[i].
    By observation, we deduce that it is always better to pick from the front
    or the back since loss[i] is increasing for 1. and is decreasing for 2.

    Assume that picking `left` chocolates from the left and `right = m - left`
    chocolates from the right is optimal. Therefore, we are selecting
    chocolates from `prices[0..left - 1]` and `prices[n - right..n - 1]`.

    To determine the optimal `left` in each iteration, we simply compare
    `loss[left]` with `loss[n - right]` if `loss[left] < loss[n - right]`,
    it's worth increasing `left`.
    """
    n = len(prices)
    countNoGreaterThanK = bisect.bisect_right(prices, k)
    l = 0
    r = min(countNoGreaterThanK, m)

    while l < r:
      mid = (l + r) // 2
      right = m - mid
      # Picking prices[mid] is better than picking prices[n - right].
      if prices[mid] < 2 * k - prices[n - right]:
        l = mid + 1
      else:
        r = mid

    return l

  def _getRelativeLoss(self, countFront: int, countBack: int, k: int, prefix: List[int]) -> int:
    """Returns the relative loss of picking `countFront` and `countBack` chocolates."""
    lossFront = prefix[countFront]
    lossBack = 2 * k * countBack - (prefix[-1] - prefix[-countBack - 1])
    return lossFront + lossBack


# Link: https://leetcode.com/problems/word-ladder/description/
class Solution:
  def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    wordSet = set(wordList)
    if endWord not in wordSet:
      return 0

    ans = 0
    q = collections.deque([beginWord])

    while q:
      ans += 1
      for _ in range(len(q)):
        wordList = list(q.popleft())
        for i, cache in enumerate(wordList):
          for c in string.ascii_lowercase:
            wordList[i] = c
            word = ''.join(wordList)
            if word == endWord:
              return ans + 1
            if word in wordSet:
              q.append(word)
              wordSet.remove(word)
          wordList[i] = cache

    return 0


# Link: https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/description/
class Solution:
  def minInsertions(self, s: str) -> int:
    return len(s) - self._longestPalindromeSubseq(s)

  # Same as 516. Longest Palindromic Subsequence
  def _longestPalindromeSubseq(self, s: str) -> int:
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


# Link: https://leetcode.com/problems/n-queens/description/
class Solution:
  def solveNQueens(self, n: int) -> List[List[str]]:
    ans = []
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def dfs(i: int, board: List[int]) -> None:
      if i == n:
        ans.append(board)
        return

      for j in range(n):
        if cols[j] or diag1[i + j] or diag2[j - i + n - 1]:
          continue
        cols[j] = diag1[i + j] = diag2[j - i + n - 1] = True
        dfs(i + 1, board + ['.' * j + 'Q' + '.' * (n - j - 1)])
        cols[j] = diag1[i + j] = diag2[j - i + n - 1] = False

    dfs(0, [])
    return ans


# Link: https://leetcode.com/problems/n-queens-ii/description/
class Solution:
  def totalNQueens(self, n: int) -> int:
    ans = 0
    cols = [False] * n
    diag1 = [False] * (2 * n - 1)
    diag2 = [False] * (2 * n - 1)

    def dfs(i: int) -> None:
      nonlocal ans
      if i == n:
        ans += 1
        return

      for j in range(n):
        if cols[j] or diag1[i + j] or diag2[j - i + n - 1]:
          continue
        cols[j] = diag1[i + j] = diag2[j - i + n - 1] = True
        dfs(i + 1)
        cols[j] = diag1[i + j] = diag2[j - i + n - 1] = False

    dfs(0)
    return ans


# Link: https://leetcode.com/problems/minimum-window-substring/description/
class Solution:
  def minWindow(self, s: str, t: str) -> str:
    count = collections.Counter(t)
    required = len(t)
    bestLeft = -1
    minLength = len(s) + 1

    l = 0
    for r, c in enumerate(s):
      count[c] -= 1
      if count[c] >= 0:
        required -= 1
      while required == 0:
        if r - l + 1 < minLength:
          bestLeft = l
          minLength = r - l + 1
        count[s[l]] += 1
        if count[s[l]] > 0:
          required += 1
        l += 1

    return '' if bestLeft == -1 else s[bestLeft: bestLeft + minLength]


# Link: https://leetcode.com/problems/maximum-strong-pair-xor-ii/description/
class TrieNode:
  def __init__(self):
    self.children: List[Optional[TrieNode]] = [None] * 2
    self.min = math.inf
    self.max = -math.inf


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
      node.min = min(node.min, num)
      node.max = max(node.max, num)

  def getMaxXor(self, x: int) -> int:
    """Returns max(x ^ y) where |x - y| <= min(x, y).

    If x <= y, |x - y| <= min(x, y) can be written as y - x <= x.
    So, y <= 2 * x.
    """
    maxXor = 0
    node = self.root
    for i in range(self.maxBit, -1, -1):
      bit = x >> i & 1
      toggleBit = bit ^ 1
      # If `node.children[toggleBit].max > x`, it means there's a number in the
      # node that satisfies the condition to ensure that x <= y among x and y.
      # If `node.children[toggleBit].min <= 2 * x`, it means there's a number in
      # the node that satisfies the condition for a valid y.
      if node.children[toggleBit] \
              and node.children[toggleBit].max > x \
              and node.children[toggleBit].min <= 2 * x:
        maxXor = maxXor | 1 << i
        node = node.children[toggleBit]
      elif node.children[bit]:
        node = node.children[bit]
      else:  # There's nothing in the Bit Trie.
        return 0
    return maxXor


class Solution:
  # Same as 2932. Maximum Strong Pair XOR I
  def maximumStrongPairXor(self, nums: List[int]) -> int:
    maxNum = max(nums)
    maxBit = int(math.log2(maxNum))
    bitTrie = BitTrie(maxBit)

    for num in nums:
      bitTrie.insert(num)

    return max(bitTrie.getMaxXor(num) for num in nums)


# Link: https://leetcode.com/problems/zuma-game/description/
class Solution:
  def findMinStep(self, board: str, hand: str) -> int:
    def deDup(board):
      start = 0  # the start index of a color sequenece
      for i, c in enumerate(board):
        if c != board[start]:
          if i - start >= 3:
            return deDup(board[:start] + board[i:])
          start = i  # Meet a new sequence.
      return board

    @functools.lru_cache(None)
    def dfs(board: str, hand: str):
      board = deDup(board)
      if board == '#':
        return 0

      boardSet = set(board)
      # hand that is in board
      hand = ''.join(h for h in hand if h in boardSet)
      if not hand:  # infeasible
        return math.inf

      ans = math.inf

      for i in range(len(board)):
        for j, h in enumerate(hand):
          # Place hs[j] in board[i].
          newHand = hand[:j] + hand[j + 1:]
          newBoard = board[:i] + h + board[i:]
          ans = min(ans, 1 + dfs(newBoard, newHand))

      return ans

    ans = dfs(board + '#', hand)
    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/stone-game-iii/description/
class Solution:
  def stoneGameIII(self, stoneValue: List[int]) -> str:
    n = len(stoneValue)
    # dp[i] := the maximum relative score Alice can make with stoneValue[i..n)
    dp = [-math.inf] * n + [0]

    for i in reversed(range(n)):
      summ = 0
      for j in range(i, i + 3):
        if j == n:
          break
        summ += stoneValue[j]
        dp[i] = max(dp[i], summ - dp[j + 1])

    score = dp[0]
    if score == 0:
      return 'Tie'
    return 'Alice' if score > 0 else 'Bob'


# Link: https://leetcode.com/problems/stone-game-iii/description/
class Solution:
  def stoneGameIII(self, stoneValue: List[int]) -> str:
    @functools.lru_cache(None)
    def dp(i: int) -> int:
      """
      Returns the maximum relative score Alice can make with stoneValue[i..n).
      """
      if i == len(stoneValue):
        return 0

      res = -math.inf
      summ = 0

      for j in range(i, i + 3):
        if j == len(stoneValue):
          break
        summ += stoneValue[j]
        res = max(res, summ - dp(j + 1))

      return res

    score = dp(0)
    if score == 0:
      return 'Tie'
    return 'Alice' if score > 0 else 'Bob'


# Link: https://leetcode.com/problems/maximize-consecutive-elements-in-an-array-after-modification/description/
class Solution:
  def maxSelectedElements(self, nums: List[int]) -> int:
    ans = 1
    prev = -math.inf
    # the length of the longest consecutive elements (seq0) ending at the
    # previous number
    dp0 = 1
    # the length of the longest consecutive elements (seq1) ending at the
    # previous number + 1
    dp1 = 1

    for num in sorted(nums):
      if num == prev:
        dp1 = dp0 + 1  # Append `num + 1` to seq0.
      elif num == prev + 1:
        dp0 += 1  # Append `num` to seq0.
        dp1 += 1  # Add 1 to every number in seq0 and append `num + 1` to seq0.
      elif num == prev + 2:
        dp0 = dp1 + 1  # Append `num` to seq1.
        dp1 = 1        # Start a new sequence [`num + 1`].
      else:
        dp0 = 1  # Start a new sequence [`num`].
        dp1 = 1  # Start a new sequence [`num + 1`].
      ans = max(ans, dp0, dp1)
      prev = num

    return ans


# Link: https://leetcode.com/problems/maximize-consecutive-elements-in-an-array-after-modification/description/
class Solution:
  def maxSelectedElements(self, nums: List[int]) -> int:
    ans = 0
    # {num: the length of the longest consecutive elements ending at num}
    dp = {}

    for num in sorted(nums):
      dp[num + 1] = dp.get(num, 0) + 1
      dp[num] = dp.get(num - 1, 0) + 1
      ans = max(ans, dp[num], dp[num + 1])

    return ans


# Link: https://leetcode.com/problems/design-a-text-editor/description/
class TextEditor:
  def __init__(self):
    self.s = []
    self.stack = []

  def addText(self, text: str) -> None:
    for c in text:
      self.s.append(c)

  def deleteText(self, k: int) -> int:
    numDeleted = min(k, len(self.s))
    for _ in range(numDeleted):
      self.s.pop()
    return numDeleted

  def cursorLeft(self, k: int) -> str:
    while self.s and k > 0:
      self.stack.append(self.s.pop())
      k -= 1
    return self._getString()

  def cursorRight(self, k: int) -> str:
    while self.stack and k > 0:
      self.s.append(self.stack.pop())
      k -= 1
    return self._getString()

  def _getString(self) -> str:
    if len(self.s) < 10:
      return ''.join(self.s)
    return ''.join(self.s[-10:])


# Link: https://leetcode.com/problems/maximum-cost-of-trip-with-k-highways/description/
class Solution:
  def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
    if k + 1 > n:
      return -1

    graph = [[] for _ in range(n)]

    for u, v, w in highways:
      graph[u].append((v, w))
      graph[v].append((u, w))

    @functools.lru_cache(None)
    def dp(u: int, mask: int) -> int:
      """
      Returns the maximum cost of trip starting from u, where `mask` is the
      bitmask of the visited cities.
      """
      if mask.bit_count() == k + 1:
        return 0

      res = -1
      for v, w in graph[u]:
        if mask >> v & 1:
          continue
        nextCost = dp(v, mask | 1 << v)
        if nextCost != -1:
          res = max(res, w + nextCost)
      return res

    return max(dp(i, 1 << i) for i in range(n))


# Link: https://leetcode.com/problems/count-complete-substrings/description/
class Solution:
  def countCompleteSubstrings(self, word: str, k: int) -> int:
    uniqueLetters = len(set(word))
    return sum(self._countCompleteStrings(word, k, windowSize)
               for windowSize in range(k, k * uniqueLetters + 1, k))

  def _countCompleteStrings(self, word: str, k: int, windowSize: int) -> int:
    """
    Returns the number of complete substrings of `windowSize` of `word`.
    """
    res = 0
    countLetters = 0  # the number of letters in the running substring
    count = collections.Counter()

    for i, c in enumerate(word):
      count[c] += 1
      countLetters += 1
      if i > 0 and abs(ord(c) - ord(word[i - 1])) > 2:
        count = collections.Counter()
        # Start a new substring starting at word[i].
        count[c] += 1
        countLetters = 1
      if countLetters == windowSize + 1:
        count[word[i - windowSize]] -= 1
        countLetters -= 1
      if countLetters == windowSize:
        res += all(freq == 0 or freq == k for freq in count.values())

    return res


# Link: https://leetcode.com/problems/count-subarrays-with-score-less-than-k/description/
class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    ans = 0
    summ = 0

    l = 0
    for r, num in enumerate(nums):
      summ += num
      while summ * (r - l + 1) >= k:
        summ -= nums[l]
        l += 1
      ans += r - l + 1

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-arrays-similar/description/
class Solution:
  def makeSimilar(self, nums: List[int], target: List[int]) -> int:
    nums.sort(key=lambda x: (x & 1, x))
    target.sort(key=lambda x: (x & 1, x))
    return sum(abs(a - b) for a, b in zip(nums, target)) // 4


# Link: https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/description/
class Solution:
  def numWays(self, steps: int, arrLen: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of ways to stay at index i
    dp = [0] * min(steps // 2 + 1, arrLen)
    dp[0] = 1

    for _ in range(steps):
      newDp = [0] * min(steps // 2 + 1, arrLen)
      for i, ways in enumerate(dp):
        if ways > 0:
          for dx in (-1, 0, 1):
            nextIndex = i + dx
            if 0 <= nextIndex < len(dp):
              newDp[nextIndex] += ways
              newDp[nextIndex] %= kMod
      dp = newDp

    return dp[0]


# Link: https://leetcode.com/problems/maximum-number-of-removal-queries-that-can-be-processed-i/description/
class Solution:
  def maximumProcessableQueries(self, nums: List[int], queries: List[int]) -> int:
    n = len(nums)
    # dp[i][j] := the maximum number of queries processed if nums[i..j] are not
    # removed after processing dp[i][j] queries
    dp = [[0] * n for _ in range(n)]

    for d in range(n - 1, -1, -1):
      for i in range(n):
        j = i + d
        if j >= n:
          continue
        if i > 0:
          # Remove nums[i - 1] from nums[i - 1..j] if possible.
          dp[i][j] = max(dp[i][j], dp[i - 1][j] +
                         (nums[i - 1] >= queries[dp[i - 1][j]]))
        if j + 1 < n:
          # Remove nums[j + 1] from nums[i..j + 1] if possible.
          dp[i][j] = max(dp[i][j], dp[i][j + 1] +
                         (nums[j + 1] >= queries[dp[i][j + 1]]))
        if dp[i][j] == len(queries):
          return len(queries)

    return max(dp[i][i] + (nums[i] >= queries[dp[i][i]])
               for i in range(n))


# Link: https://leetcode.com/problems/minimum-changes-to-make-k-semi-palindromes/description/
class Solution:
  def minimumChanges(self, s: str, k: int) -> int:
    n = len(s)
    # factors[i] := factors of i
    factors = self._getFactors(n)
    # cost[i][j] := changes to make s[i..j] a semi-palindrome
    cost = self._getCost(s, n, factors)
    # dp[i][j] := the minimum changes to split s[i:] into j valid parts
    dp = [[n] * (k + 1) for _ in range(n + 1)]

    dp[n][0] = 0

    for i in range(n - 1, -1, -1):
      for j in range(1, k + 1):
        for l in range(i + 1, n):
          dp[i][j] = min(dp[i][j], dp[l + 1][j - 1] + cost[i][l])

    return dp[0][k]

  def _getFactors(self, n: int) -> List[List[int]]:
    factors = [[1] for _ in range(n + 1)]
    for d in range(2, n):
      for i in range(d * 2, n + 1, d):
        factors[i].append(d)
    return factors

  def _getCost(self, s: str, n: int, factors: List[List[int]]) -> List[List[int]]:
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
      for j in range(i + 1, n):
        length = j - i + 1
        minCost = length
        for d in factors[length]:
          minCost = min(minCost, self._getCostD(s, i, j, d))
        cost[i][j] = minCost
    return cost

  def _getCostD(self, s: str, i: int, j: int, d: int) -> int:
    """Returns the cost to make s[i..j] a semi-palindrome of `d`."""
    cost = 0
    for offset in range(d):
      l = i + offset
      r = j - d + 1 + offset
      while l < r:
        if s[l] != s[r]:
          cost += 1
        l += d
        r -= d
    return cost


# Link: https://leetcode.com/problems/construct-target-array-with-multiple-sums/description/
class Solution:
  def isPossible(self, target: List[int]) -> bool:
    if len(target) == 1:
      return target[0] == 1

    summ = sum(target)
    maxHeap = [-num for num in target]
    heapq.heapify(maxHeap)

    while -maxHeap[0] > 1:
      maxi = -heapq.heappop(maxHeap)
      restSum = summ - maxi
      # Only occurs if n == 2.
      if restSum == 1:
        return True
      updated = maxi % restSum
      # updated == 0 (invalid) or didn't change.
      if updated == 0 or updated == maxi:
        return False
      heapq.heappush(maxHeap, -updated)
      summ = summ - maxi + updated

    return True


# Link: https://leetcode.com/problems/abbreviating-the-product-of-a-range/description/
class Solution:
  def abbreviateProduct(self, left: int, right: int) -> str:
    prod = 1.0
    suf = 1
    countDigits = 0
    countZeros = 0

    for num in range(left, right + 1):
      prod *= num
      while prod >= 1.0:
        prod /= 10
        countDigits += 1
      suf *= num
      while suf % 10 == 0:
        suf //= 10
        countZeros += 1
      if suf > 10 ** 8:
        suf %= 10 ** 8

    if countDigits - countZeros <= 10:
      tens = 10 ** (countDigits - countZeros)
      return str(int(prod * tens + 0.5)) + 'e' + str(countZeros)

    pre = str(int(prod * 10 ** 5))
    suf = str(suf)[-5:]
    return pre + '...' + suf + 'e' + str(countZeros)


# Link: https://leetcode.com/problems/cat-and-mouse-ii/description/
class Solution:
  def canMouseWin(self, grid: List[str], catJump: int, mouseJump: int) -> bool:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    nFloors = 0
    cat = 0  # cat's position
    mouse = 0  # mouse's position

    def hash(i: int, j: int) -> int:
      return i * n + j

    for i in range(m):
      for j in range(n):
        if grid[i][j] != '#':
          nFloors += 1
        if grid[i][j] == 'C':
          cat = hash(i, j)
        elif grid[i][j] == 'M':
          mouse = hash(i, j)

    @functools.lru_cache(None)
    def dp(cat: int, mouse: int, turn: int) -> bool:
      """
      Returns True if the mouse can win, where the cat is on (i / 8, i % 8), the
      mouse is on (j / 8, j % 8), and the turns is k.
      """
      # We already search the whole touchable grid.
      if turn == nFloors * 2:
        return False

      if turn % 2 == 0:
        # the mouse's turn
        i = mouse // n
        j = mouse % n
        for dx, dy in dirs:
          for jump in range(mouseJump + 1):
            x = i + dx * jump
            y = j + dy * jump
            if x < 0 or x == m or y < 0 or y == n:
              break
            if grid[x][y] == '#':
              break
            # The mouse eats the food, so the mouse wins.
            if grid[x][y] == 'F':
              return True
            if dp(cat, hash(x, y), turn + 1):
              return True
        # The mouse can't win, so the mouse loses.
        return False
      else:
        # the cat's turn
        i = cat // n
        j = cat % n
        for dx, dy in dirs:
          for jump in range(catJump + 1):
            x = i + dx * jump
            y = j + dy * jump
            if x < 0 or x == m or y < 0 or y == n:
              break
            if grid[x][y] == '#':
              break
            # The cat eats the food, so the mouse loses.
            if grid[x][y] == 'F':
              return False
            nextCat = hash(x, y)
            # The cat catches the mouse, so the mouse loses.
            if nextCat == mouse:
              return False
            if not dp(nextCat, mouse, turn + 1):
              return False
        # The cat can't win, so the mouse wins.
        return True

    return dp(cat, mouse, 0)


# Link: https://leetcode.com/problems/max-dot-product-of-two-subsequences/description/
class Solution:
  def maxDotProduct(self, A: List[int], B: List[int]) -> int:
    m = len(A)
    n = len(B)
    # dp[i][j] := the maximum dot product of the two subsequences nums[0..i)
    # and nums2[0..j)
    dp = [[-math.inf] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
      for j in range(n):
        dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j],
                               max(0, dp[i][j]) + A[i] * B[j])

    return dp[m][n]


# Link: https://leetcode.com/problems/number-of-paths-with-max-score/description/
class Solution:
  def pathsWithMaxScore(self, board: List[str]) -> List[int]:
    kMod = 1_000_000_007
    n = len(board)
    dirs = ((0, 1), (1, 0), (1, 1))
    # dp[i][j] := the maximum sum from (n - 1, n - 1) to (i, j)
    dp = [[-1] * (n + 1) for _ in range(n + 1)]
    # count[i][j] := the number of paths to get dp[i][j] from (n - 1, n - 1) to
    # (i, j)
    count = [[0] * (n + 1) for _ in range(n + 1)]

    dp[0][0] = 0
    dp[n - 1][n - 1] = 0
    count[n - 1][n - 1] = 1

    for i in reversed(range(n)):
      for j in reversed(range(n)):
        if board[i][j] == 'S' or board[i][j] == 'X':
          continue
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if dp[i][j] < dp[x][y]:
            dp[i][j] = dp[x][y]
            count[i][j] = count[x][y]
          elif dp[i][j] == dp[x][y]:
            count[i][j] += count[x][y]
            count[i][j] %= kMod

        # If there's path(s) from 'S' to (i, j) and the cell is not 'E'.
        if dp[i][j] != -1 and board[i][j] != 'E':
          dp[i][j] += int(board[i][j])
          dp[i][j] %= kMod

    return [dp[0][0], count[0][0]]


# Link: https://leetcode.com/problems/modify-graph-edge-weights/description/

class Solution:
  def modifiedGraphEdges(self, n: int, edges: List[List[int]], source: int, destination: int, target: int) -> List[List[int]]:
    kMax = 2_000_000_000
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
      if w == -1:
        continue
      graph[u].append((v, w))
      graph[v].append((u, w))

    distToDestination = self._dijkstra(graph, source, destination)
    if distToDestination < target:
      return []
    if distToDestination == target:
      # Change the weights of negative edges to an impossible value.
      for edge in edges:
        if edge[2] == -1:
          edge[2] = kMax
      return edges

    for i, (u, v, w) in enumerate(edges):
      if w != -1:
        continue
      edges[i][2] = 1
      graph[u].append((v, 1))
      graph[v].append((u, 1))
      distToDestination = self._dijkstra(graph, source, destination)
      if distToDestination <= target:
        edges[i][2] += target - distToDestination
        # Change the weights of negative edges to an impossible value.
        for j in range(i + 1, len(edges)):
          if edges[j][2] == -1:
            edges[j][2] = kMax
        return edges

    return []

  def _dijkstra(self, graph: List[List[int]], src: int, dst: int) -> int:
    dist = [math.inf] * len(graph)
    minHeap = []  # (d, u)

    dist[src] = 0
    heapq.heappush(minHeap, (dist[src], src))

    while minHeap:
      d, u = heapq.heappop(minHeap)
      for v, w in graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (dist[v], v))

    return dist[dst]


# Link: https://leetcode.com/problems/count-k-subsequences-of-a-string-with-maximum-beauty/description/
class Solution:
  def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
    kMod = 1_000_000_007
    count = collections.Counter(s)
    if len(count) < k:
      return 0

    ans = 1
    # freqCount := (f(c), # of chars with f(c))
    freqCount = collections.Counter(count.values())

    for fc, numOfChars in list(sorted(freqCount.items(), reverse=True)):
      if numOfChars >= k:
        ans *= math.comb(numOfChars, k) * pow(fc, k, kMod)
        return ans % kMod
      ans *= pow(fc, numOfChars, kMod)
      ans %= kMod
      k -= numOfChars


# Link: https://leetcode.com/problems/erect-the-fence-ii/description/
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
  x: float
  y: float


@dataclass(frozen=True)
class Disk:
  center: Point
  radius: float


class Solution:
  def outerTrees(self, trees: List[List[int]]) -> List[float]:
    points = [Point(x, y) for x, y in trees]
    disk = self._welzl(points, 0, [])
    return [disk.center.x, disk.center.y, disk.radius]

  def _welzl(self, points: List[Point], i: int, planePoints: List[Point]) -> Disk:
    """Returns the smallest disk that encloses points[i..n).

    https://en.wikipedia.org/wiki/Smallest-disk_problem#Welzl's_algorithm
    """
    if i == len(points) or len(planePoints) == 3:
      return self._trivial(planePoints)
    disk = self._welzl(points, i + 1, planePoints)
    if self._inside(disk, points[i]):
      return disk
    return self._welzl(points, i + 1, planePoints + [points[i]])

  def _trivial(self, planePoints: List[Point]) -> Disk:
    """Returns the smallest disk that encloses `planePoints`."""
    if len(planePoints) == 0:
      return Disk(Point(0, 0), 0)
    if len(planePoints) == 1:
      return Disk(Point(planePoints[0].x, planePoints[0].y), 0)
    if len(planePoints) == 2:
      return self._getDisk(planePoints[0], planePoints[1])

    disk01 = self._getDisk(planePoints[0], planePoints[1])
    if self._inside(disk01, planePoints[2]):
      return disk01

    disk02 = self._getDisk(planePoints[0], planePoints[2])
    if self._inside(disk02, planePoints[1]):
      return disk02

    disk12 = self._getDisk(planePoints[1], planePoints[2])
    if self._inside(disk12, planePoints[0]):
      return disk12

    return self._getDiskFromThree(planePoints[0], planePoints[1], planePoints[2])

  def _getDisk(self, A: Point, B: Point) -> Disk:
    """Returns the smallest disk that encloses the points A and B."""
    x = (A.x + B.x) / 2
    y = (A.y + B.y) / 2
    return Disk(Point(x, y), self._distance(A, B) / 2)

  def _getDiskFromThree(self, A: Point, B: Point, C: Point) -> Disk:
    """Returns the smallest disk that encloses the points A, B, and C."""
    # Calculate midpoints.
    mAB = Point((A.x + B.x) / 2, (A.y + B.y) / 2)
    mBC = Point((B.x + C.x) / 2, (B.y + C.y) / 2)

    # Calculate the slopes and the perpendicular slopes.
    slopeAB = math.inf if B.x == A.x else (B.y - A.y) / (B.x - A.x)
    slopeBC = math.inf if C.x == B.x else (C.y - B.y) / (C.x - B.x)
    perpSlopeAB = math.inf if slopeAB == 0 else -1 / slopeAB
    perpSlopeBC = math.inf if slopeBC == 0 else -1 / slopeBC

    # Calculate the center.
    x = (perpSlopeBC * mBC.x - perpSlopeAB * mAB.x +
         mAB.y - mBC.y) / (perpSlopeBC - perpSlopeAB)
    y = perpSlopeAB * (x - mAB.x) + mAB.y
    center = Point(x, y)
    return Disk(center, self._distance(center, A))

  def _inside(self, disk: Disk, point: Point) -> bool:
    """Returns True if the point is inside the disk."""
    return disk.radius > 0 and self._distance(disk.center, point) <= disk.radius

  def _distance(self, A: Point, B: Point) -> float:
    dx = A.x - B.x
    dy = A.y - B.y
    return math.sqrt(dx**2 + dy**2)


# Link: https://leetcode.com/problems/verbal-arithmetic-puzzle/description/
class Solution:
  def isSolvable(self, words: List[str], result: str) -> bool:
    words.append(result)
    rows = len(words)
    cols = max(map(len, words))
    letterToDigit = {}
    usedDigit = [False] * 10

    def dfs(row: int, col: int, summ: int) -> bool:
      if col == cols:
        return summ == 0
      if row == rows:
        return summ % 10 == 0 and dfs(0, col + 1, summ // 10)

      word = words[row]
      if col >= len(word):
        return dfs(row + 1, col, summ)

      letter = word[~col]
      sign = -1 if row == rows - 1 else 1

      if letter in letterToDigit and (letterToDigit[letter] > 0 or col < len(word) - 1):
        return dfs(row + 1, col, summ + sign * letterToDigit[letter])

      for digit, used in enumerate(usedDigit):
        if not used and (digit > 0 or col < len(word) - 1):
          letterToDigit[letter] = digit
          usedDigit[digit] = True
          if dfs(row + 1, col, summ + sign * digit):
            return True
          usedDigit[digit] = False
          if letter in letterToDigit:
            del letterToDigit[letter]

      return False

    return dfs(0, 0, 0)


# Link: https://leetcode.com/problems/subsequence-with-the-minimum-score/description/
class Solution:
  def minimumScore(self, s: str, t: str) -> int:
    # leftmost[j] := the minimum index i s.t. t[0..j] is a subsequence of s[0..i].
    #          -1 := impossible
    leftmost = [-1] * len(t)
    # rightmost[j] := the maximum index i s.t. t[j:] is a subsequence of s[i..n).
    #           -1 := impossible
    rightmost = [-1] * len(t)

    j = 0  # t's index
    for i in range(len(s)):
      if s[i] == t[j]:
        leftmost[j] = i
        j += 1
        if j == len(t):
          break

    j = len(t) - 1  # t's index
    for i in reversed(range(len(s))):
      if s[i] == t[j]:
        rightmost[j] = i
        j -= 1
        if j == -1:
          break

    # The worst case is to delete t[0:j] since t[j:] is a subsequence of s. (deduced
    # from the above loop).
    ans = j + 1

    j = 0
    for i in range(len(t)):
      # It's impossible that t[0..i] is a subsequence of s. So, stop the loop since
      # no need to consider any larger i.
      if leftmost[i] == -1:
        break
      # While t[0..i] + t[j:] is not a subsequence of s, increase j.
      while j < len(t) and leftmost[i] >= rightmost[j]:
        j += 1
      # Now, leftmost[i] < rightmost[j], so t[0..i] + t[j:n] is a subsequence of s.
      # If i == j that means t is a subsequence of s, so just return 0.
      if i == j:
        return 0
      # Delete t[i + 1..j - 1] and that's a total of j - i - 1 letters.
      ans = min(ans, j - i - 1)

    return ans


# Link: https://leetcode.com/problems/find-longest-awesome-substring/description/
class Solution:
  def longestAwesome(self, s: str) -> int:
    ans = 0
    prefix = 0  # the binary prefix
    prefixToIndex = [len(s)] * 1024
    prefixToIndex[0] = -1

    for i, c in enumerate(s):
      prefix ^= 1 << ord(c) - ord('0')
      ans = max(ans, i - prefixToIndex[prefix])
      for j in range(10):
        ans = max(ans, i - prefixToIndex[prefix ^ 1 << j])
      prefixToIndex[prefix] = min(prefixToIndex[prefix], i)

    return ans


# Link: https://leetcode.com/problems/number-of-subarrays-that-match-a-pattern-ii/description/
class Solution:
  # Same as 3034. Number of Subarrays That Match a Pattern I
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


# Link: https://leetcode.com/problems/contains-duplicate-iii/description/
class Solution:
  def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    if not nums or indexDiff <= 0 or valueDiff < 0:
      return False

    mini = min(nums)
    diff = valueDiff + 1  # In case that `valueDiff` equals 0.
    bucket = {}

    def getKey(num: int) -> int:
      return (num - mini) // diff

    for i, num in enumerate(nums):
      key = getKey(num)
      if key in bucket:  # the current bucket
        return True
      # the left adjacent bucket
      if key - 1 in bucket and num - bucket[key - 1] < diff:
        return True
      # the right adjacent bucket
      if key + 1 in bucket and bucket[key + 1] - num < diff:
        return True
      bucket[key] = num
      if i >= indexDiff:
        del bucket[getKey(nums[i - indexDiff])]

    return False


# Link: https://leetcode.com/problems/minimize-max-distance-to-gas-station/description/
class Solution:
  def minmaxGasDist(self, stations: List[int], k: int) -> float:
    kErr = 1e-6
    l = 0
    r = stations[-1] - stations[0]

    def possible(k: int, m: float) -> bool:
      """
      Returns true if can use <= k gas stations to ensure that each adjacent
      distance between gas stations <= m.
      """
      for a, b in zip(stations, stations[1:]):
        diff = b - a
        if diff > m:
          k -= math.ceil(diff / m) - 1
          if k < 0:
            return False
      return True

    while r - l > kErr:
      m = (l + r) / 2
      if possible(k, m):
        r = m
      else:
        l = m

    return l


# Link: https://leetcode.com/problems/design-video-sharing-platform/description/
class VideoSharingPlatform:
  def __init__(self):
    self.currVideoId = 0
    self.usedIds = []
    self.videoIdToVideo = {}
    self.videoIdToViews = collections.Counter()
    self.videoIdToLikes = collections.Counter()
    self.videoIdToDislikes = collections.Counter()

  def upload(self, video: str) -> int:
    videoId = self._getVideoId()
    self.videoIdToVideo[videoId] = video
    return videoId

  def remove(self, videoId: int) -> None:
    if videoId in self.videoIdToVideo:
      heapq.heappush(self.usedIds, videoId)
      del self.videoIdToVideo[videoId]
      del self.videoIdToViews[videoId]
      del self.videoIdToLikes[videoId]
      del self.videoIdToDislikes[videoId]

  def watch(self, videoId: int, startMinute: int, endMinute: int) -> str:
    if videoId not in self.videoIdToVideo:
      return '-1'
    self.videoIdToViews[videoId] += 1
    video = self.videoIdToVideo[videoId]
    return video[startMinute:min(endMinute + 1, len(video))]

  def like(self, videoId: int) -> None:
    if videoId in self.videoIdToVideo:
      self.videoIdToLikes[videoId] += 1

  def dislike(self, videoId: int) -> None:
    if videoId in self.videoIdToVideo:
      self.videoIdToDislikes[videoId] += 1

  def getLikesAndDislikes(self, videoId: int) -> List[int]:
    if videoId in self.videoIdToVideo:
      return [self.videoIdToLikes[videoId], self.videoIdToDislikes[videoId]]
    return [-1]

  def getViews(self, videoId: int) -> int:
    if videoId in self.videoIdToVideo:
      return self.videoIdToViews[videoId]
    return -1

  def _getVideoId(self) -> int:
    if not self.usedIds:
      self.currVideoId += 1
      return self.currVideoId - 1
    return heapq.heappop(self.usedIds)


# Link: https://leetcode.com/problems/minimum-time-to-finish-the-race/description/
class Solution:
  def minimumFinishTime(self, tires: List[List[int]], changeTime: int, numLaps: int) -> int:
    # singleTire[i] := the minimum time to finish i laps without changing tire
    singleTire = [math.inf] * (numLaps + 1)
    # dp[i] := the minimum time to finish i laps
    dp = [math.inf] * (numLaps + 1)

    for i, (f, r) in enumerate(tires):
      sumSecs = 0
      rPower = 1
      for j in range(1, numLaps + 1):
        # the time to use the same tire for the next lap >=
        # the time to change a new tire + f
        if f * rPower >= changeTime + f:
          break
        sumSecs += f * rPower
        rPower *= r
        singleTire[j] = min(singleTire[j], sumSecs)

    dp[0] = 0
    for i in range(1, numLaps + 1):
      for j in range(1, i + 1):
        dp[i] = min(dp[i], dp[i - j] + changeTime + singleTire[j])

    return dp[numLaps] - changeTime


# Link: https://leetcode.com/problems/sum-of-scores-of-built-strings/description/
class Solution:
  def sumScores(self, s: str) -> int:
    n = len(s)
    # https://cp-algorithms.com/string/z-function.html#implementation
    z = [0] * n
    # [l, r] := the indices of the rightmost segment match
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

    return sum(z) + n


# Link: https://leetcode.com/problems/shortest-impossible-sequence-of-rolls/description/
class Solution:
  def shortestSequence(self, rolls: List[int], k: int) -> int:
    ans = 1  # the the next target length
    seen = set()

    for roll in rolls:
      seen.add(roll)
      if len(seen) == k:
        # Have all combinations that form `ans` length, and we are going to
        # extend the sequence to `ans + 1` length.
        ans += 1
        seen.clear()

    return ans


# Link: https://leetcode.com/problems/the-skyline-problem/description/
class Solution:
  def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
    n = len(buildings)
    if n == 0:
      return []
    if n == 1:
      left, right, height = buildings[0]
      return [[left, height], [right, 0]]

    left = self.getSkyline(buildings[:n // 2])
    right = self.getSkyline(buildings[n // 2:])
    return self._merge(left, right)

  def _merge(self, left: List[List[int]], right: List[List[int]]) -> List[List[int]]:
    ans = []
    i = 0  # left's index
    j = 0  # right's index
    leftY = 0
    rightY = 0

    while i < len(left) and j < len(right):
      # Choose the powith smaller x
      if left[i][0] < right[j][0]:
        leftY = left[i][1]  # Update the ongoing `leftY`.
        self._addPoint(ans, left[i][0], max(left[i][1], rightY))
        i += 1
      else:
        rightY = right[j][1]  # Update the ongoing `rightY`.
        self._addPoint(ans, right[j][0], max(right[j][1], leftY))
        j += 1

    while i < len(left):
      self._addPoint(ans, left[i][0], left[i][1])
      i += 1

    while j < len(right):
      self._addPoint(ans, right[j][0], right[j][1])
      j += 1

    return ans

  def _addPoint(self, ans: List[List[int]], x: int, y: int) -> None:
    if ans and ans[-1][0] == x:
      ans[-1][1] = y
      return
    if ans and ans[-1][1] == y:
      return
    ans.append([x, y])


# Link: https://leetcode.com/problems/create-components-with-same-value/description/
class Solution:
  def componentValue(self, nums: List[int], edges: List[List[int]]) -> int:
    kMax = 1_000_000_000
    n = len(nums)
    summ = sum(nums)
    tree = [[] for _ in range(n)]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int, target: int, seen: Set[bool]) -> int:
      """
      Returns the sum of the subtree rooted at u substracting the sum of the
      deleted subtrees.
      """
      summ = nums[u]
      seen.add(u)

      for v in tree[u]:
        if v in seen:
          continue
        summ += dfs(v, target, seen)
        if summ > target:
          return kMax

      # Delete the tree that has sum == target.
      if summ == target:
        return 0
      return summ

    for i in range(n, 1, -1):
      # Split the tree into i parts, i.e. delete (i - 1) edges.
      if summ % i == 0 and dfs(0, summ // i, set()) == 0:
        return i - 1

    return 0


# Link: https://leetcode.com/problems/minimum-cost-to-convert-string-ii/description/
class Solution:
  def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
    subLengths = set(len(s) for s in original)
    subToId = self._getSubToId(original, changed)
    subCount = len(subToId)
    # dist[u][v] := the minimum distance to change the substring with id u to
    # the substring with id v
    dist = [[math.inf for _ in range(subCount)] for _ in range(subCount)]
    # dp[i] := the minimum cost to change the first i letters of `source` into
    # `target`, leaving the suffix untouched
    dp = [math.inf for _ in range(len(source) + 1)]

    for a, b, c in zip(original, changed, cost):
      u = subToId[a]
      v = subToId[b]
      dist[u][v] = min(dist[u][v], c)

    for k in range(subCount):
      for i in range(subCount):
        if dist[i][k] < math.inf:
          for j in range(subCount):
            if dist[k][j] < math.inf:
              dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    dp[0] = 0

    for i, (s, t) in enumerate(zip(source, target)):
      if dp[i] == math.inf:
        continue
      if s == t:
        dp[i + 1] = min(dp[i + 1], dp[i])
      for subLength in subLengths:
        if i + subLength > len(source):
          continue
        subSource = source[i:i + subLength]
        subTarget = target[i:i + subLength]
        if subSource not in subToId or subTarget not in subToId:
          continue
        u = subToId[subSource]
        v = subToId[subTarget]
        if dist[u][v] != math.inf:
          dp[i + subLength] = min(dp[i + subLength], dp[i] + dist[u][v])

    return -1 if dp[len(source)] == math.inf else dp[len(source)]

  def _getSubToId(self, original: str, changed: str) -> Dict[str, int]:
    subToId = {}
    for s in original + changed:
      if s not in subToId:
        subToId[s] = len(subToId)
    return subToId


# Link: https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/description/
from enum import Enum


class BoxCase(Enum):
  kEqualDistantBalls = 0
  kEqualBalls = 1


class Solution:
  def getProbability(self, balls: List[int]) -> float:
    n = sum(balls) // 2
    fact = [1, 1, 2, 6, 24, 120, 720]

    def cases(i: int, ballsCountA: int, ballsCountB: int,
              colorsCountA: int, colorsCountB, boxCase: BoxCase) -> float:
      if ballsCountA > n or ballsCountB > n:
        return 0
      if i == len(balls):
        return 1 if boxCase == BoxCase.kEqualBalls else colorsCountA == colorsCountB

      ans = 0.0

      # balls taken from A for `balls[i]`
      for ballsTakenA in range(balls[i] + 1):
        ballsTakenB = balls[i] - ballsTakenA
        newcolorsCountA = colorsCountA + (ballsTakenA > 0)
        newcolorsCountB = colorsCountB + (ballsTakenB > 0)
        ans += cases(i + 1, ballsCountA + ballsTakenA, ballsCountB + ballsTakenB,
                     newcolorsCountA, newcolorsCountB, boxCase) / \
            (fact[ballsTakenA] * fact[ballsTakenB])

      return ans

    return cases(0, 0, 0, 0, 0, BoxCase.kEqualDistantBalls) / \
        cases(0, 0, 0, 0, 0, BoxCase.kEqualBalls)


# Link: https://leetcode.com/problems/optimal-account-balancing/description/
class Solution:
  def minTransfers(self, transactions: List[List[int]]) -> int:
    balance = [0] * 21

    for u, v, amount in transactions:
      balance[u] -= amount
      balance[v] += amount

    debts = [b for b in balance if b]

    def dfs(s: int) -> int:
      while s < len(debts) and not debts[s]:
        s += 1
      if s == len(debts):
        return 0

      ans = math.inf

      for i in range(s + 1, len(debts)):
        if debts[i] * debts[s] < 0:
          debts[i] += debts[s]  # `debts[s]` is settled.
          ans = min(ans, 1 + dfs(s + 1))
          debts[i] -= debts[s]  # Backtrack.

      return ans

    return dfs(0)


# Link: https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/
class Solution:
  def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
    n = len(jobDifficulty)
    if d > n:
      return -1

    # dp[i][k] := the minimum difficulty to schedule the first i jobs in k days
    dp = [[math.inf] * (d + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
      for k in range(1, d + 1):
        maxDifficulty = 0  # max(job[j + 1..i])
        for j in range(i - 1, k - 2, -1):  # 1-based
          maxDifficulty = max(maxDifficulty, jobDifficulty[j])  # 0-based
          dp[i][k] = min(dp[i][k], dp[j][k - 1] + maxDifficulty)

    return dp[n][d]


# Link: https://leetcode.com/problems/strobogrammatic-number-iii/description/
class Solution:
  def strobogrammaticInRange(self, low: str, high: str) -> int:
    pairs = [['0', '0'], ['1', '1'], ['6', '9'], ['8', '8'], ['9', '6']]
    ans = 0

    def dfs(s: List[chr], l: int, r: int) -> None:
      nonlocal ans
      if l > r:
        if len(s) == len(low) and ''.join(s) < low:
          return
        if len(s) == len(high) and ''.join(s) > high:
          return
        ans += 1
        return

      for leftDigit, rightDigit in pairs:
        if l == r and leftDigit != rightDigit:
          continue
        s[l] = leftDigit
        s[r] = rightDigit
        if len(s) > 1 and s[0] == '0':
          continue
        dfs(s, l + 1, r - 1)

    for n in range(len(low), len(high) + 1):
      dfs([' '] * n, 0, n - 1)

    return ans


# Link: https://leetcode.com/problems/maximum-strictly-increasing-cells-in-a-matrix/description/
class Solution:
  def maxIncreasingCells(self, mat: List[List[int]]) -> int:
    m = len(mat)
    n = len(mat[0])
    rows = [0] * m  # rows[i] := the maximum path length for the i-th row
    cols = [0] * n  # cols[j] := the maximum path length for the j-th column
    valToIndices = collections.defaultdict(list)
    # maxPathLength[i][j] := the maximum path length from mat[i][j]
    maxPathLength = [[0] * n for _ in range(m)]
    # Sort all the unique values in the matrix in non-increasing order.
    decreasingSet = set()

    for i in range(m):
      for j in range(n):
        val = mat[i][j]
        valToIndices[val].append((i, j))
        decreasingSet.add(val)

    for val in sorted(decreasingSet, reverse=True):
      for i, j in valToIndices[val]:
        maxPathLength[i][j] = max(rows[i], cols[j]) + 1
      for i, j in valToIndices[val]:
        rows[i] = max(rows[i], maxPathLength[i][j])
        cols[j] = max(cols[j], maxPathLength[i][j])

    return max(max(rows), max(cols))


# Link: https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/description/
class Solution:
  def minimumTime(self, grid: List[List[int]]) -> int:
    if grid[0][1] > 1 and grid[1][0] > 1:
      return -1

    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    minHeap = [(0, 0, 0)]  # (time, i, j)
    seen = {(0, 0)}

    while minHeap:
      time, i, j = heapq.heappop(minHeap)
      if i == m - 1 and j == n - 1:
        return time
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if (x, y) in seen:
          continue
        extraWait = 1 if (grid[x][y] - time) % 2 == 0 else 0
        nextTime = max(time + 1, grid[x][y] + extraWait)
        heapq.heappush(minHeap, (nextTime, x, y))
        seen.add((x, y))


# Link: https://leetcode.com/problems/total-appeal-of-a-string/description/
class Solution:
  def appealSum(self, s: str) -> int:
    ans = 0
    lastSeen = {}

    for i, c in enumerate(s):
      ans += (i - lastSeen.get(c, -1)) * (len(s) - i)
      lastSeen[c] = i

    return ans


# Link: https://leetcode.com/problems/total-appeal-of-a-string/description/
class Solution:
  def appealSum(self, s: str) -> int:
    ans = 0
    # the total appeal of all substrings ending in the index so far
    dp = 0
    lastSeen = {}

    for i, c in enumerate(s):
      #   the total appeal of all substrings ending in s[i]
      # = the total appeal of all substrings ending in s[i - 1]
      # + the number of substrings ending in s[i] that contain only this s[i]
      dp += i - lastSeen.get(c, -1)
      ans += dp
      lastSeen[c] = i

    return ans


# Link: https://leetcode.com/problems/number-of-great-partitions/description/
class Solution:
  def countPartitions(self, nums: List[int], k: int) -> int:
    kMod = 1_000_000_007
    summ = sum(nums)
    ans = pow(2, len(nums), kMod)  # 2^n % kMod
    dp = [1] + [0] * k

    for num in nums:
      for i in range(k, num - 1, -1):
        dp[i] += dp[i - num]
        dp[i] %= kMod

    # Substract the cases that're not satisfied.
    for i in range(k):
      if summ - i < k:  # Both group1 and group2 < k.
        ans -= dp[i]
      else:
        ans -= dp[i] * 2

    return ans % kMod


# Link: https://leetcode.com/problems/count-array-pairs-divisible-by-k/description/
class Solution:
  def countPairs(self, nums: List[int], k: int) -> int:
    ans = 0
    gcds = collections.Counter()

    for num in nums:
      gcd_i = math.gcd(num, k)
      for gcd_j, count in gcds.items():
        if gcd_i * gcd_j % k == 0:
          ans += count
      gcds[gcd_i] += 1

    return ans


# Link: https://leetcode.com/problems/number-of-ways-to-earn-points/description/
class Solution:
  def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
    kMod = 1_000_000_007
    # dp[i][j] := the number of ways to earn j points with the first i types
    dp = [[0] * (target + 1) for _ in range(len(types) + 1)]
    dp[0][0] = 1

    for i in range(1, len(types) + 1):
      count = types[i - 1][0]
      mark = types[i - 1][1]
      for j in range(target + 1):
        for solved in range(count + 1):
          if j - solved * mark >= 0:
            dp[i][j] += dp[i - 1][j - solved * mark]
            dp[i][j] %= kMod

    return dp[len(types)][target]


# Link: https://leetcode.com/problems/number-of-ways-to-earn-points/description/
class Solution:
  def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
    kMod = 1_000_000_007
    # dp[j] := the number of ways to earn j points with the types so far
    dp = [1] + [0] * target

    for count, mark in types:
      for j in range(target, -1, -1):
        for solved in range(1, count + 1):
          if j - solved * mark >= 0:
            dp[j] += dp[j - solved * mark]
            dp[j] %= kMod

    return dp[target]


# Link: https://leetcode.com/problems/maximum-number-of-visible-points/description/
class Solution:
  def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
    posX, posY = location
    maxVisible = 0
    same = 0
    A = []

    for x, y in points:
      if x == posX and y == posY:
        same += 1
      else:
        A.append(math.atan2(y - posY, x - posX))

    A.sort()
    A = A + [a + 2.0 * math.pi for a in A]

    angleInRadians = math.pi * (angle / 180)

    l = 0
    for r in range(len(A)):
      while A[r] - A[l] > angleInRadians:
        l += 1
      maxVisible = max(maxVisible, r - l + 1)

    return maxVisible + same


# Link: https://leetcode.com/problems/encrypt-and-decrypt-strings/description/
class Encrypter:
  def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
    self.keyToValue = {k: v for k, v in zip(keys, values)}
    self.decrypt = collections.Counter(self.encrypt(word)
                                       for word in dictionary).__getitem__

  def encrypt(self, word1: str) -> str:
    return ''.join(self.keyToValue[c] for c in word1)


# Link: https://leetcode.com/problems/encrypt-and-decrypt-strings/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = collections.defaultdict(TrieNode)
    self.isWord = False


class Encrypter:
  def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
    self.keyToValue = {k: v for k, v in zip(keys, values)}
    self.valueToKeys = collections.defaultdict(list)
    self.root = TrieNode()
    for k, v in zip(keys, values):
      self.valueToKeys[v].append(k)
    for word in dictionary:
      self._insert(word)

  def encrypt(self, word1: str) -> str:
    return ''.join(self.keyToValue[c] for c in word1)

  def decrypt(self, word2: str) -> int:
    return self._find(word2, 0, self.root)

  def _insert(self, word: str) -> None:
    node = self.root
    for c in word:
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True

  def _find(self, word: str, i: int, node: TrieNode) -> int:
    value = word[i:i + 2]
    if value not in self.valueToKeys:
      return 0

    ans = 0
    if i + 2 == len(word):
      for key in self.valueToKeys[value]:
        ans += node.children[key].isWord
      return ans

    for key in self.valueToKeys[value]:
      if key not in node.children:
        continue
      ans += self._find(word, i + 2, node.children[key])

    return ans


# Link: https://leetcode.com/problems/distribute-repeating-integers/description/
class Solution:
  def canDistribute(self, nums: List[int], quantity: List[int]) -> bool:
    freqs = list(collections.Counter(nums).values())
    # validDistribution[i][j] := True if it's possible to distribute the i-th
    # freq into a subset of quantity represented by the bitmask j
    validDistribution = self._getValidDistribution(freqs, quantity)
    n = len(freqs)
    m = len(quantity)
    maxMask = 1 << m
    # dp[i][j] := true if it's possible to distribute freqs[i..n), where j is
    # the bitmask of the selected quantity
    dp = [[False] * maxMask for _ in range(n + 1)]
    dp[n][maxMask - 1] = True

    for i in range(n - 1, -1, -1):
      for mask in range(maxMask):
        dp[i][mask] = dp[i + 1][mask]
        availableMask = ~mask & (maxMask - 1)
        submask = availableMask
        while submask > 0:
          if validDistribution[i][submask]:
            dp[i][mask] = dp[i][mask] or dp[i + 1][mask | submask]
          submask = (submask - 1) & availableMask

    return dp[0][0]

  def _getValidDistribution(self, freqs: List[int], quantity: List[int]) -> List[List[bool]]:
    maxMask = 1 << len(quantity)
    validDistribution = [[False] * maxMask for _ in range(len(freqs))]
    for i, freq in enumerate(freqs):
      for mask in range(maxMask):
        if freq >= self._getQuantitySum(quantity, mask):
          validDistribution[i][mask] = True
    return validDistribution

  def _getQuantitySum(self, quantity: List[int], mask: int) -> int:
    """Returns the sum of the selected quantity represented by `mask`."""
    return sum(q for i, q in enumerate(quantity) if mask >> i & 1)


# Link: https://leetcode.com/problems/word-ladder-ii/description/
class Solution:
  def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wordSet = set(wordList)
    if endWord not in wordList:
      return []

    # {"hit": ["hot"], "hot": ["dot", "lot"], ...}
    graph: Dict[str, List[str]] = collections.defaultdict(list)

    # Build the graph from the beginWord to the endWord.
    if not self._bfs(beginWord, endWord, wordSet, graph):
      return []

    ans = []

    self._dfs(graph, beginWord, endWord, [beginWord], ans)
    return ans

  def _bfs(self, beginWord: str, endWord: str, wordSet: Set[str], graph: Dict[str, List[str]]) -> bool:
    q1 = {beginWord}
    q2 = {endWord}
    backward = False

    while q1 and q2:
      for word in q1:
        wordSet.discard(word)
      for word in q2:
        wordSet.discard(word)
      # Always expand the smaller queue.
      if len(q1) > len(q2):
        q1, q2 = q2, q1
        backward = not backward
      q = set()
      reachEndWord = False
      for parent in q1:
        for child in self._getChildren(parent, wordSet, q2):
          if child in wordSet or child in q2:
            q.add(child)
            if backward:
              graph[child].append(parent)
            else:
              graph[parent].append(child)
          if child in q2:
            reachEndWord = True
      if reachEndWord:
        return True
      q1 = q

    return False

  def _getChildren(self, parent: str, wordSet: Set[str], q2) -> List[str]:
    children = []
    s = list(parent)

    for i, cache in enumerate(s):
      for c in string.ascii_lowercase:
        if c == cache:
          continue
        s[i] = c
        child = ''.join(s)
        if child in wordSet or child in q2:
          children.append(child)
      s[i] = cache

    return children

  def _dfs(self, graph: Dict[str, List[str]], word: str, endWord: str, path: List[str], ans: List[List[str]]) -> None:
    if word == endWord:
      ans.append(path.copy())
      return

    for child in graph.get(word, []):
      path.append(child)
      self._dfs(graph, child, endWord, path, ans)
      path.pop()


# Link: https://leetcode.com/problems/word-ladder-ii/description/
class Solution:
  def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    wordSet = set(wordList)
    if endWord not in wordList:
      return []

    # {"hit": ["hot"], "hot": ["dot", "lot"], ...}
    graph: Dict[str, List[str]] = collections.defaultdict(list)

    # Build the graph from the beginWord to the endWord.
    if not self._bfs(beginWord, endWord, wordSet, graph):
      return []

    ans = []
    self._dfs(graph, beginWord, endWord, [beginWord], ans)
    return ans

  def _bfs(self, beginWord: str, endWord: str, wordSet: Set[str], graph: Dict[str, List[str]]) -> bool:
    currentLevelWords = {beginWord}

    while currentLevelWords:
      for word in currentLevelWords:
        wordSet.discard(word)
      nextLevelWords = set()
      reachEndWord = False
      for parent in currentLevelWords:
        for child in self._getChildren(parent, wordSet):
          if child in wordSet:
            nextLevelWords.add(child)
            graph[parent].append(child)
          if child == endWord:
            reachEndWord = True
      if reachEndWord:
        return True
      currentLevelWords = nextLevelWords

    return False

  def _getChildren(self, parent: str, wordSet: Set[str]) -> List[str]:
    children = []
    s = list(parent)

    for i, cache in enumerate(s):
      for c in string.ascii_lowercase:
        if c == cache:
          continue
        s[i] = c
        child = ''.join(s)
        if child in wordSet:
          children.append(child)
      s[i] = cache

    return children

  def _dfs(self, graph: Dict[str, List[str]], word: str, endWord: str, path: List[str], ans: List[List[str]]) -> None:
    if word == endWord:
      ans.append(path.copy())
      return

    for child in graph.get(word, []):
      path.append(child)
      self._dfs(graph, child, endWord, path, ans)
      path.pop()


# Link: https://leetcode.com/problems/find-all-good-strings/description/
class Solution:
  def findGoodStrings(self, n: int, s1: str, s2: str, evil: str) -> int:
    kMod = 1_000_000_007
    evilLPS = self._getLPS(evil)

    @functools.lru_cache(None)
    def getNextMatchedEvilCount(j: int, currChar: str) -> int:
      """
      Returns the number of next matched evil count, where there're j matches
      with `evil` and the current letter is ('a' + j).
      """
      while j > 0 and evil[j] != currChar:
        j = evilLPS[j - 1]
      return j + 1 if evil[j] == currChar else j

    @functools.lru_cache(None)
    def dp(i: int, matchedEvilCount: int, isS1Prefix: bool, isS2Prefix: bool) -> int:
      """
      Returns the number of good strings for s[i..n), where there're j matches
      with `evil`, `isS1Prefix` indicates if the current letter is tightly bound
      for `s1` and `isS2Prefix` indicates if the current letter is tightly bound
      for `s2`.
      """
      # s[0..i) contains `evil`, so don't consider any ongoing strings.
      if matchedEvilCount == len(evil):
        return 0
      # Run out of strings, so contribute one.
      if i == n:
        return 1
      ans = 0
      minCharIndex = ord(s1[i]) if isS1Prefix else ord('a')
      maxCharIndex = ord(s2[i]) if isS2Prefix else ord('z')
      for charIndex in range(minCharIndex, maxCharIndex + 1):
        c = chr(charIndex)
        nextMatchedEvilCount = getNextMatchedEvilCount(matchedEvilCount, c)
        ans += dp(i + 1, nextMatchedEvilCount,
                  isS1Prefix and c == s1[i],
                  isS2Prefix and c == s2[i])
        ans %= kMod
      return ans

    return dp(0, 0, True, True)

  def _getLPS(self, pattern: str) -> List[int]:
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


# Link: https://leetcode.com/problems/minimum-total-cost-to-make-arrays-unequal/description/
class Solution:
  def minimumTotalCost(self, nums1: List[int], nums2: List[int]) -> int:
    n = len(nums1)
    ans = 0
    maxFreq = 0
    maxFreqNum = 0
    shouldBeSwapped = 0
    conflictedNumCount = [0] * (n + 1)

    # Collect the indices i s.t. num1 == num2 and record their `maxFreq`
    # and `maxFreqNum`.
    for i, (num1, num2) in enumerate(zip(nums1, nums2)):
      if num1 == num2:
        conflictedNum = num1
        conflictedNumCount[conflictedNum] += 1
        if conflictedNumCount[conflictedNum] > maxFreq:
          maxFreq = conflictedNumCount[conflictedNum]
          maxFreqNum = conflictedNum
        shouldBeSwapped += 1
        ans += i

    # Collect the indices with num1 != num2 that contribute less cost.
    # This can be greedily achieved by iterating from 0 to n - 1.
    for i, (num1, num2) in enumerate(zip(nums1, nums2)):
      # Since we have over `maxFreq * 2` spaces, `maxFreqNum` can be
      # successfully distributed, so no need to collectextra spaces.
      if maxFreq * 2 <= shouldBeSwapped:
        break
      if num1 == num2:
        continue
      # The numbers == `maxFreqNum` worsen the result since they increase the
      # `maxFreq`.
      if num1 == maxFreqNum or num2 == maxFreqNum:
        continue
      shouldBeSwapped += 1
      ans += i

    return -1 if maxFreq * 2 > shouldBeSwapped else ans


# Link: https://leetcode.com/problems/number-of-ways-to-reconstruct-a-tree/description/
class Solution:
  def checkWays(self, pairs: List[List[int]]) -> int:
    kMax = 501
    graph = collections.defaultdict(list)
    degrees = [0] * kMax
    connected = [[False] * kMax for _ in range(kMax)]

    for u, v in pairs:
      graph[u].append(v)
      graph[v].append(u)
      degrees[u] += 1
      degrees[v] += 1
      connected[u][v] = True
      connected[v][u] = True

    # For each node, sort its children by degrees in descending order.
    for _, children in graph.items():
      children.sort(key=lambda a: degrees[a], reverse=True)

    # Find the root with a degree that equals to n - 1.
    root = next((i for i, d in enumerate(degrees) if d == len(graph) - 1), -1)
    if root == -1:
      return 0

    hasMoreThanOneWay = False

    # Returns true if each node rooted at u is connected to all of its ancestors.
    def dfs(u: int, ancestors: List[int], seen: List[bool]) -> bool:
      nonlocal hasMoreThanOneWay
      seen[u] = True
      for ancestor in ancestors:
        if not connected[u][ancestor]:
          return False
      ancestors.append(u)
      for v in graph[u]:
        if seen[v]:
          continue
        if degrees[v] == degrees[u]:
          hasMoreThanOneWay = True
        if not dfs(v, ancestors, seen):
          return False
      ancestors.pop()
      return True

    if not dfs(root, [], [False] * kMax):
      return 0
    return 2 if hasMoreThanOneWay else 1


# Link: https://leetcode.com/problems/maximize-score-after-n-operations/description/
class Solution:
  def maxScore(self, nums: List[int]) -> int:
    n = len(nums) // 2

    @functools.lru_cache(None)
    def dp(k: int, mask: int) -> int:
      """
      Returns the maximum score you can receive after performing the k to n
      operations, where `mask` is the bitmask of the chosen numbers.
      """
      if k == n + 1:
        return 0

      res = 0

      for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
          chosenMask = 1 << i | 1 << j
          if (mask & chosenMask) == 0:
            res = max(res,
                      k * math.gcd(nums[i], nums[j]) + dp(k + 1, mask | chosenMask))

      return res

    return dp(1, 0)


# Link: https://leetcode.com/problems/maximum-frequency-score-of-a-subarray/description/
class Solution:
  def maxFrequencyScore(self, nums: List[int], k: int) -> int:
    kMod = 1_000_000_007
    count = collections.Counter(nums[:k])
    summ = self._getInitialSumm(count, kMod)
    ans = summ

    for i in range(k, len(nums)):
      # Remove the leftmost number that's out-of-window.
      leftNum = nums[i - k]
      summ = (summ - pow(leftNum, count[leftNum], kMod) + kMod) % kMod
      # After decreasing its frequency, if it's still > 0, then add it back.
      count[leftNum] -= 1
      if count[leftNum] > 0:
        summ = (summ + pow(leftNum, count[leftNum], kMod)) % kMod
      # Otherwise, remove it from the count map.
      else:
        del count[leftNum]
      # Add the current number. Similarly, remove the current score like above.
      rightNum = nums[i]
      if count[rightNum] > 0:
        summ = (summ - pow(rightNum, count[rightNum], kMod) + kMod) % kMod
      count[rightNum] += 1
      summ = (summ + pow(rightNum, count[rightNum], kMod)) % kMod
      ans = max(ans, summ)

    return ans

  def _getInitialSumm(self, count: Dict[int, int], kMod: int) -> int:
    summ = 0
    for num, freq in count.items():
      summ = (summ + pow(num, freq, kMod)) % kMod
    return summ


# Link: https://leetcode.com/problems/handshakes-that-dont-cross/description/
class Solution:
  def numberOfWays(self, numPeople: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of ways i handshakes could occure s.t. none of the
    # handshakes cross
    dp = [1] + [0] * (numPeople // 2)

    for i in range(1, numPeople // 2 + 1):
      for j in range(i):
        dp[i] += dp[j] * dp[i - 1 - j]
        dp[i] %= kMod

    return dp[numPeople // 2]


# Link: https://leetcode.com/problems/minimize-deviation-in-array/description/
class Solution:
  def minimumDeviation(self, nums: List[int]) -> int:
    ans = math.inf
    mini = math.inf
    maxHeap = []

    for num in nums:
      evenNum = num if num % 2 == 0 else num * 2
      heapq.heappush(maxHeap, -evenNum)
      mini = min(mini, evenNum)

    while maxHeap[0] % 2 == 0:
      maxi = -heapq.heappop(maxHeap)
      ans = min(ans, maxi - mini)
      mini = min(mini, maxi // 2)
      heapq.heappush(maxHeap, -maxi // 2)

    return min(ans, -maxHeap[0] - mini)


# Link: https://leetcode.com/problems/rectangle-area-ii/description/
class Solution:
  def rectangleArea(self, rectangles: List[List[int]]) -> int:
    events = []

    for x1, y1, x2, y2 in rectangles:
      events.append((x1, y1, y2, 's'))
      events.append((x2, y1, y2, 'e'))

    events.sort(key=lambda x: x[0])

    ans = 0
    prevX = 0
    yPairs = []

    def getHeight(yPairs: List[Tuple[int, int]]) -> int:
      height = 0
      prevY = 0

      for y1, y2 in yPairs:
        prevY = max(prevY, y1)
        if y2 > prevY:
          height += y2 - prevY
          prevY = y2

      return height

    for currX, y1, y2, type in events:
      if currX > prevX:
        width = currX - prevX
        ans += width * getHeight(yPairs)
        prevX = currX
      if type == 's':
        yPairs.append((y1, y2))
        yPairs.sort()
      else:  # type == 'e'
        yPairs.remove((y1, y2))

    return ans % (10**9 + 7)


# Link: https://leetcode.com/problems/move-sub-tree-of-n-ary-tree/description/
class Solution:
  def moveSubTree(self, root: 'Node', p: 'Node', q: 'Node') -> 'Node':
    if p in q.children:
      return root

    # Create a dummy Node for the case when root == p
    dummy = Node(None, [root])

    # Get each parent of p and q
    pParent = self._getParent(dummy, p)
    qParent = self._getParent(p, q)

    # Get p's original index in p's parent
    pIndex = pParent.children.index(p)
    pParent.children.pop(pIndex)

    q.children.append(p)

    # If q is in the p's subtree, qParent != None
    if qParent:
      qParent.children.remove(q)
      pParent.children.insert(pIndex, q)

    return dummy.children[0]

  def _getParent(self, root: 'Node', target: 'Node') -> Optional['Node']:
    for child in root.children:
      if child == target:
        return root
      res = self._getParent(child, target)
      if res:
        return res
    return None


# Link: https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/description/
class IndexedQuery:
  def __init__(self, queryIndex: int, query: int):
    self.queryIndex = queryIndex
    self.query = query

  def __iter__(self):
    yield self.queryIndex
    yield self.query


class Solution:
  def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    ans = [0] * len(queries)
    minHeap = [(grid[0][0], 0, 0)]  # (grid[i][j], i, j)
    seen = {(0, 0)}
    accumulate = 0

    for queryIndex, query in sorted([IndexedQuery(i, query)
                                     for i, query in enumerate(queries)],
                                    key=lambda iq: iq.query):
      while minHeap:
        val, i, j = heapq.heappop(minHeap)
        if val >= query:
          # The smallest neighbor is still larger than `query`, so no need to
          # keep exploring. Re-push (i, j, grid[i][j]) back to the `minHeap`.
          heapq.heappush(minHeap, (val, i, j))
          break
        accumulate += 1
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if (x, y) in seen:
            continue
          heapq.heappush(minHeap, (grid[x][y], x, y))
          seen.add((x, y))
      ans[queryIndex] = accumulate

    return ans


# Link: https://leetcode.com/problems/valid-palindrome-iii/description/
class Solution:
  def isValidPalindrome(self, s: str, k: int) -> bool:
    return len(s) - self._longestPalindromeSubseq(s) <= k

  # Same as 516. Longest Palindromic Subsequence
  def _longestPalindromeSubseq(self, s: str) -> int:
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


# Link: https://leetcode.com/problems/rank-transform-of-a-matrix/description/
class UnionFind:
  def __init__(self):
    self.id = {}

  def union(self, u: int, v: int) -> None:
    self.id.setdefault(u, u)
    self.id.setdefault(v, v)
    i = self._find(u)
    j = self._find(v)
    if i != j:
      self.id[i] = j

  def getGroupIdToValues(self) -> Dict[int, List[int]]:
    groupIdToValues = collections.defaultdict(list)
    for u in self.id.keys():
      groupIdToValues[self._find(u)].append(u)
    return groupIdToValues

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
    m = len(matrix)
    n = len(matrix[0])
    ans = [[0] * n for _ in range(m)]
    # {val: [(i, j)]}
    valToGrids = collections.defaultdict(list)
    # rank[i] := the maximum rank of the row or column so far
    maxRankSoFar = [0] * (m + n)

    for i, row in enumerate(matrix):
      for j, val in enumerate(row):
        valToGrids[val].append((i, j))

    for _, grids in sorted(valToGrids.items()):
      uf = UnionFind()
      for i, j in grids:
        # Union i-th row with j-th col.
        uf.union(i, j + m)
      for values in uf.getGroupIdToValues().values():
        # Get the maximum rank of all the included rows and columns.
        maxRank = max(maxRankSoFar[i] for i in values)
        for i in values:
          # Update all the rows and columns to maxRank + 1.
          maxRankSoFar[i] = maxRank + 1
      for i, j in grids:
        ans[i][j] = maxRankSoFar[i]

    return ans


# Link: https://leetcode.com/problems/the-earliest-and-latest-rounds-where-players-compete/description/
class Solution:
  def earliestAndLatest(self, n: int,
                        firstPlayer: int, secondPlayer: int) -> List[int]:
    @functools.lru_cache(None)
    def dp(l: int, r: int, k: int) -> List[int]:
      """
      Returns the (earliest, latest) pair, the first player is the l-th player
      from the front, the second player is the r-th player from the end, and
      there're k people.
      """
      if l == r:
        return [1, 1]
      if l > r:
        return dp(r, l, k)

      a = math.inf
      b = -math.inf

      # Enumerate all the possible positions.
      for i in range(1, l + 1):
        for j in range(l - i + 1, r - i + 1):
          if not l + r - k // 2 <= i + j <= (k + 1) // 2:
            continue
          x, y = dp(i, j, (k + 1) // 2)
          a = min(a, x + 1)
          b = max(b, y + 1)

      return [a, b]

    return dp(firstPlayer, n - secondPlayer + 1, n)


# Link: https://leetcode.com/problems/gcd-sort-of-an-array/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> None:
    i = self.find(u)
    j = self.find(v)
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

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def gcdSort(self, nums: List[int]) -> bool:
    maxNum = max(nums)
    minPrimeFactors = self._sieveEratosthenes(maxNum + 1)
    uf = UnionFind(maxNum + 1)

    for num in nums:
      for primeFactor in self._getPrimeFactors(num, minPrimeFactors):
        uf.unionByRank(num, primeFactor)

    for a, b in zip(nums, sorted(nums)):
      # Can't swap nums[i] with sortedNums[i].
      if uf.find(a) != uf.find(b):
        return False

    return True

  def _sieveEratosthenes(self, n: int) -> List[int]:
    """Gets the minimum prime factor of i, where 1 < i <= n."""
    minPrimeFactors = [i for i in range(n + 1)]
    for i in range(2, int(n**0.5) + 1):
      if minPrimeFactors[i] == i:  # `i` is prime.
        for j in range(i * i, n, i):
          minPrimeFactors[j] = min(minPrimeFactors[j], i)
    return minPrimeFactors

  def _getPrimeFactors(self, num: int, minPrimeFactors: List[int]) -> List[int]:
    primeFactors = []
    while num > 1:
      divisor = minPrimeFactors[num]
      primeFactors.append(divisor)
      while num % divisor == 0:
        num //= divisor
    return primeFactors


# Link: https://leetcode.com/problems/jump-game-v/description/
class Solution:
  def maxJumps(self, arr: List[int], d: int) -> int:
    n = len(arr)
    # dp[i] := the maximum jumps starting from arr[i]
    dp = [1] * n
    # a dcreasing stack that stores indices
    stack = []

    for i in range(n + 1):
      while stack and (i == n or arr[stack[-1]] < arr[i]):
        indices = [stack.pop()]
        while stack and arr[stack[-1]] == arr[indices[0]]:
          indices.append(stack.pop())
        for j in indices:
          if i < n and i - j <= d:
            # Can jump from i to j.
            dp[i] = max(dp[i], dp[j] + 1)
          if stack and j - stack[-1] <= d:
            # Can jump from stack[-1] to j
            dp[stack[-1]] = max(dp[stack[-1]], dp[j] + 1)
      stack.append(i)

    return max(dp)


# Link: https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/description/
class Solution:
  def minTaps(self, n: int, ranges: List[int]) -> int:
    nums = [0] * (n + 1)

    for i, range_ in enumerate(ranges):
      l = max(0, i - range_)
      r = min(n, range_ + i)
      nums[l] = max(nums[l], r - l)

    ans = 0
    end = 0
    farthest = 0

    for i in range(n):
      farthest = max(farthest, i + nums[i])
      if i == end:
        ans += 1
        end = farthest

    return ans if end == n else -1


# Link: https://leetcode.com/problems/robot-collisions/description/
@dataclass
class Robot:
  index: int
  position: int
  health: int
  direction: str


class Solution:
  def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
    robots = sorted([Robot(index, position, health, direction)
                     for index, (position, health, direction) in
                     enumerate(zip(positions, healths, directions))],
                    key=lambda robot: robot.position)
    stack: List[Robot] = []  # running robots

    for robot in robots:
      if robot.direction == 'R':
        stack.append(robot)
        continue
      # Collide with robots going right if any.
      while stack and stack[-1].direction == 'R' and robot.health > 0:
        if stack[-1].health == robot.health:
          stack.pop()
          robot.health = 0
        elif stack[-1].health < robot.health:
          stack.pop()
          robot.health -= 1
        else:  # stack[-1].health > robot.health
          stack[-1].health -= 1
          robot.health = 0
      if robot.health > 0:
        stack.append(robot)

    stack.sort(key=lambda robot: robot.index)
    return [robot.health for robot in stack]


# Link: https://leetcode.com/problems/reverse-nodes-in-k-group/description/
class Solution:
  def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head:
      return None

    tail = head

    for _ in range(k):
      # There are less than k nodes in the list, do nothing.
      if not tail:
        return head
      tail = tail.next

    newHead = self._reverse(head, tail)
    head.next = self.reverseKGroup(tail, k)
    return newHead

  def _reverse(self, head: Optional[ListNode], tail: Optional[ListNode]) -> Optional[ListNode]:
    """Reverses [head, tail)."""
    prev = None
    curr = head
    while curr != tail:
      next = curr.next
      curr.next = prev
      prev = curr
      curr = next
    return prev


# Link: https://leetcode.com/problems/reverse-nodes-in-k-group/description/
class Solution:
  def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    if not head or k == 1:
      return head

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

    for _ in range(length // k):
      for _ in range(k - 1):
        next = curr.next
        curr.next = next.next
        next.next = prev.next
        prev.next = next
      prev = curr
      curr = curr.next

    return dummy.next


# Link: https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/description/
class Solution:
  def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
    ans = 0
    maxRight = max(startPos, fruits[-1][0])
    amounts = [0] * (1 + maxRight)
    for position, amount in fruits:
      amounts[position] = amount
    prefix = [0] + list(itertools.accumulate(amounts))

    def getFruits(leftSteps: int, rightSteps: int) -> int:
      l = max(0, startPos - leftSteps)
      r = min(maxRight, startPos + rightSteps)
      return prefix[r + 1] - prefix[l]

    # Go right first.
    for rightSteps in range(min(maxRight - startPos, k) + 1):
      leftSteps = max(0, k - 2 * rightSteps)  # Turn left
      ans = max(ans, getFruits(leftSteps, rightSteps))

    # Go left first.
    for leftSteps in range(min(startPos, k) + 1):
      rightSteps = max(0, k - 2 * leftSteps)  # Turn right
      ans = max(ans, getFruits(leftSteps, rightSteps))

    return ans


# Link: https://leetcode.com/problems/subarrays-with-k-different-integers/description/
class Solution:
  def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
    def subarraysWithAtMostKDistinct(k: int) -> int:
      ans = 0
      count = collections.Counter()

      l = 0
      for r, num in enumerate(nums):
        count[num] += 1
        if count[num] == 1:
          k -= 1
        while k < 0:
          count[nums[l]] -= 1
          if count[nums[l]] == 0:
            k += 1
          l += 1
        ans += r - l + 1  # nums[l..r], nums[l + 1..r], ..., nums[r]

      return ans

    return subarraysWithAtMostKDistinct(k) - subarraysWithAtMostKDistinct(k - 1)


# Link: https://leetcode.com/problems/minimum-cost-to-split-an-array/description/
class Solution:
  def minCost(self, nums: List[int], k: int) -> int:
    kMax = 1001
    n = len(nums)
    # trimmedLength[i][j] := trimmed(nums[i..j]).length
    trimmedLength = [[0] * n for _ in range(n)]
    # dp[i] := the minimum cost to split nums[i..n)
    dp = [math.inf] * n + [0]

    for i in range(n):
      length = 0
      count = [0] * kMax
      for j in range(i, n):
        count[nums[j]] += 1
        if count[nums[j]] == 2:
          length += 2
        elif count[nums[j]] > 2:
          length += 1
        trimmedLength[i][j] = length

    dp[n] = 0

    for i in range(n - 1, -1, -1):
      for j in range(i, n):
        dp[i] = min(dp[i], k + trimmedLength[i][j] + dp[j + 1])

    return dp[0]


# Link: https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/
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
  def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
    criticalEdges = []
    pseudoCriticalEdges = []

    # Record the index information, so edges[i] := (u, v, weight, index).
    for i in range(len(edges)):
      edges[i].append(i)

    # Sort by the weight.
    edges.sort(key=lambda x: x[2])

    def getMSTWeight(firstEdge: List[int], deletedEdgeIndex: int) -> Union[int, float]:
      mstWeight = 0
      uf = UnionFind(n)

      if firstEdge:
        uf.unionByRank(firstEdge[0], firstEdge[1])
        mstWeight += firstEdge[2]

      for u, v, weight, index in edges:
        if index == deletedEdgeIndex:
          continue
        if uf.find(u) == uf.find(v):
          continue
        uf.unionByRank(u, v)
        mstWeight += weight

      root = uf.find(0)
      if any(uf.find(i) != root for i in range(n)):
        return math.inf

      return mstWeight

    mstWeight = getMSTWeight([], -1)

    for edge in edges:
      index = edge[3]
      # Deleting the `edge` increases the weight of the MST or makes the MST
      # invalid.
      if getMSTWeight([], index) > mstWeight:
        criticalEdges.append(index)
      # If an edge can be in any MST, we can always add `edge` to the edge set.
      elif getMSTWeight(edge, -1) == mstWeight:
        pseudoCriticalEdges.append(index)

    return [criticalEdges, pseudoCriticalEdges]


# Link: https://leetcode.com/problems/create-maximum-number/description/
class Solution:
  def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
    def maxArray(nums: List[int], k: int) -> List[int]:
      res = []
      toTop = len(nums) - k
      for num in nums:
        while res and res[-1] < num and toTop > 0:
          res.pop()
          toTop -= 1
        res.append(num)
      return res[:k]

    def merge(nums1: List[int], nums2: List[int]) -> List[int]:
      return [max(nums1, nums2).pop(0) for _ in nums1 + nums2]

    return max(merge(maxArray(nums1, i), maxArray(nums2, k - i))
               for i in range(k + 1)
               if i <= len(nums1) and k - i <= len(nums2))


# Link: https://leetcode.com/problems/minimum-replacements-to-sort-the-array/description/
class Solution:
  def minimumReplacement(self, nums: List[int]) -> int:
    ans = 0

    max = nums[-1]
    for i in range(len(nums) - 2, -1, -1):
      ops = (nums[i] - 1) // max
      ans += ops
      max = nums[i] // (ops + 1)

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/description/
class Point:
  def __init__(self, x: float, y: float):
    self.x = x
    self.y = y


class Solution:
  def numPoints(self, darts: List[List[int]], r: int) -> int:
    kErr = 1e-6
    ans = 1
    points = [Point(x, y) for x, y in darts]

    def dist(p: Point, q: Point) -> float:
      return ((p.x - q.x)**2 + (p.y - q.y)**2)**0.5

    def getCircles(p: Point, q: Point) -> List[Point]:
      if dist(p, q) - 2.0 * r > kErr:
        return []
      m = Point((p.x + q.x) / 2, (p.y + q.y) / 2)
      distCM = (r**2 - (dist(p, q) / 2)**2)**0.5
      alpha = math.atan2(p.y - q.y, q.x - p.x)
      return [Point(m.x - distCM * math.sin(alpha), m.y - distCM * math.cos(alpha)),
              Point(m.x + distCM * math.sin(alpha), m.y + distCM * math.cos(alpha))]

    for i in range(len(points)):
      for j in range(i + 1, len(points)):
        for c in getCircles(points[i], points[j]):
          count = 0
          for point in points:
            if dist(c, point) - r <= kErr:
              count += 1
          ans = max(ans, count)

    return ans


# Link: https://leetcode.com/problems/maximum-performance-of-a-team/description/
class Solution:
  # Similar to 857. Minimum Cost to Hire K Workers
  def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
    kMod = 1_000_000_007
    ans = 0
    speedSum = 0
    # (efficiency[i], speed[i]) sorted by efficiency[i] in descending order
    A = sorted([(e, s) for s, e in zip(speed, efficiency)], reverse=True)
    minHeap = []

    for e, s in A:
      heapq.heappush(minHeap, s)
      speedSum += s
      if len(minHeap) > k:
        speedSum -= heapq.heappop(minHeap)
      ans = max(ans, speedSum * e)

    return ans % kMod


# Link: https://leetcode.com/problems/dinner-plate-stacks/description/
class DinnerPlates:
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.stacks = []
    self.minHeap = [0]  # the minimum indices of the stacks to push

  def push(self, val: int) -> None:
    index = self.minHeap[0]
    # Add a new stack on demand.
    if index == len(self.stacks):
      self.stacks.append([])
    # Push the new value.
    self.stacks[index].append(val)
    # If the stack pushed is full, remove its candidacy from `minHeap`.
    if len(self.stacks[index]) == self.capacity:
      heapq.heappop(self.minHeap)
      # If `minHeap` is empty, the next available stack index is |stacks|.
      if not self.minHeap:
        heapq.heappush(self.minHeap, len(self.stacks))

  def pop(self) -> int:
    # Remove empty stacks from the back.
    while self.stacks and not self.stacks[-1]:
      self.stacks.pop()
    if not self.stacks:
      return -1
    return self.popAtStack(len(self.stacks) - 1)

  def popAtStack(self, index: int) -> int:
    if index >= len(self.stacks) or not self.stacks[index]:
      return -1
    # If the stack is going to have space, add its candiday to `minHeap`.
    if len(self.stacks[index]) == self.capacity:
      heapq.heappush(self.minHeap, index)
    return self.stacks[index].pop()


# Link: https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/
class Solution:
  def minDays(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])

    def dfs(grid: List[List[int]], i: int, j: int, seen: Set[Tuple[int, int]]):
      seen.add((i, j))
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if grid[x][y] == 0 or (x, y) in seen:
          continue
        dfs(grid, x, y, seen)

    def disconnected(grid: List[List[int]]) -> bool:
      islandsCount = 0
      seen = set()
      for i in range(m):
        for j in range(n):
          if grid[i][j] == 0 or (i, j) in seen:
            continue
          if islandsCount > 1:
            return True
          islandsCount += 1
          dfs(grid, i, j, seen)
      return islandsCount != 1

    if disconnected(grid):
      return 0

    # Try to remove 1 land.
    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1:
          grid[i][j] = 0
          if disconnected(grid):
            return 1
          grid[i][j] = 1

    # Remove 2 lands.
    return 2


# Link: https://leetcode.com/problems/distribute-elements-into-two-arrays-ii/description/
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
  def resultArray(self, nums: List[int]) -> List[int]:
    arr1 = []
    arr2 = []
    ranks = self._getRanks(nums)
    tree1 = FenwickTree(len(ranks))
    tree2 = FenwickTree(len(ranks))

    def add(num: int, arr: List[int], tree: FenwickTree) -> None:
      arr.append(num)
      tree.update(ranks[num], 1)

    add(nums[0], arr1, tree1)
    add(nums[1], arr2, tree2)

    for i in range(2, len(nums)):
      greaterCount1 = len(arr1) - tree1.get(ranks[nums[i]])
      greaterCount2 = len(arr2) - tree2.get(ranks[nums[i]])
      if greaterCount1 > greaterCount2:
        add(nums[i], arr1, tree1)
      elif greaterCount1 < greaterCount2:
        add(nums[i], arr2, tree2)
      elif len(arr1) > len(arr2):
        add(nums[i], arr2, tree2)
      else:
        add(nums[i], arr1, tree1)

    return arr1 + arr2

  def _getRanks(self, nums: List[int]) -> Dict[int, int]:
    ranks = collections.Counter()
    rank = 0
    for num in sorted(set(nums)):
      rank += 1
      ranks[num] = rank
    return ranks


# Link: https://leetcode.com/problems/maximum-profitable-triplets-with-increasing-prices-ii/description/
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
  # Same as 2907. Maximum Profitable Triplets With Increasing Prices I
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


# Link: https://leetcode.com/problems/sudoku-solver/description/
class Solution:
  def solveSudoku(self, board: List[List[str]]) -> None:
    def isValid(row: int, col: int, c: str) -> bool:
      for i in range(9):
        if board[i][col] == c or \
           board[row][i] == c or \
           board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
          return False
      return True

    def solve(s: int) -> bool:
      if s == 81:
        return True

      i = s // 9
      j = s % 9

      if board[i][j] != '.':
        return solve(s + 1)

      for c in string.digits[1:]:
        if isValid(i, j, c):
          board[i][j] = c
          if solve(s + 1):
            return True
          board[i][j] = '.'

      return False

    solve(0)


# Link: https://leetcode.com/problems/time-to-cross-a-bridge/description/
class Solution:
  def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:
    ans = 0
    # (leftToRight + rightToLeft, i)
    leftBridgeQueue = [(-leftToRight - rightToLeft, -i)
                       for i, (leftToRight, pickOld, rightToLeft, pickNew) in enumerate(time)]
    rightBridgeQueue = []
    # (time to be idle, i)
    leftWorkers = []
    rightWorkers = []

    heapq.heapify(leftBridgeQueue)

    while n > 0 or rightBridgeQueue or rightWorkers:
      # Idle left workers get on the left bridge.
      while leftWorkers and leftWorkers[0][0] <= ans:
        i = heapq.heappop(leftWorkers)[1]
        leftWorkers.pop()
        heapq.heappush(leftBridgeQueue, (-time[i][0] - time[i][2], -i))
      # Idle right workers get on the right bridge.
      while rightWorkers and rightWorkers[0][0] <= ans:
        i = heapq.heappop(rightWorkers)[1]
        heapq.heappush(rightBridgeQueue, (-time[i][0] - time[i][2], -i))
      if rightBridgeQueue:
        # If the bridge is free, the worker waiting on the right side of the
        # bridge gets to cross the bridge. If more than one worker is waiting
        # on the right side, the one with the lowest efficiency crosses first.
        i = -heapq.heappop(rightBridgeQueue)[1]
        ans += time[i][2]
        heapq.heappush(leftWorkers, (ans + time[i][3], i))
      elif leftBridgeQueue and n > 0:
        # If the bridge is free and no worker is waiting on the right side, and
       # at least one box remains at the old warehouse, the worker on the left
       # side of the river gets to cross the bridge. If more than one worker
       # is waiting on the left side, the one with the lowest efficiency
       # crosses first.
        i = -heapq.heappop(leftBridgeQueue)[1]
        ans += time[i][0]
        heapq.heappush(rightWorkers, (ans + time[i][1], i))
        n -= 1
      else:
        # Advance the time of the last crossing worker.
        ans = min(leftWorkers[0][0] if leftWorkers and n > 0 else math.inf,
                  rightWorkers[0][0] if rightWorkers else math.inf)

    return ans


# Link: https://leetcode.com/problems/remove-boxes/description/
class Solution:
  def removeBoxes(self, boxes: List[int]) -> int:
    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> int:
      """
      Returns the maximum score of boxes[i..j] if k boxes equal to boxes[j].
      """
      if i > j:
        return 0

      r = j
      sameBoxes = k + 1
      while r > 0 and boxes[r - 1] == boxes[r]:
        r -= 1
        sameBoxes += 1
      res = dp(i, r - 1, 0) + sameBoxes * sameBoxes

      for p in range(i, r):
        if boxes[p] == boxes[r]:
          res = max(res, dp(i, p, sameBoxes) + dp(p + 1, r - 1, 0))

      return res

    return dp(0, len(boxes) - 1, 0)


# Link: https://leetcode.com/problems/word-abbreviation/description/
class IndexedWord:
  def __init__(self, word: str, index: int):
    self.word = word
    self.index = index


class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = collections.defaultdict(TrieNode)
    self.count = 0


class Solution:
  def wordsAbbreviation(self, words: List[str]) -> List[str]:
    n = len(words)
    ans = [''] * n

    def getAbbrev(s: str, prefixIndex: int) -> str:
      n = len(s)
      num = n - (prefixIndex + 1) - 1
      numLength = 1 if num < 10 else (2 if num < 100 else 3)
      abbrevLength = (prefixIndex + 1) + numLength + 1
      if abbrevLength >= n:
        return s
      return s[:prefixIndex + 1] + str(num) + s[-1]

    abbrevToIndexedWords = collections.defaultdict(list)

    for i, word in enumerate(words):
      abbrev = getAbbrev(word, 0)
      abbrevToIndexedWords[abbrev].append(IndexedWord(word, i))

    def insertWord(root: Optional[TrieNode], word: str) -> None:
      node = root
      for c in word:
        node = node.children.setdefault(c, TrieNode())
        node.count += 1

    def firstUniqueIndex(root: Optional[TrieNode], word: str) -> None:
      node = root
      for i, c in enumerate(word):
        node = node.children[c]
        if node.count == 1:
          return i
      return len(word)

    for indexedWords in abbrevToIndexedWords.values():
      root = TrieNode()
      for iw in indexedWords:
        insertWord(root, iw.word)
      for iw in indexedWords:
        index = firstUniqueIndex(root, iw.word)
        ans[iw.index] = getAbbrev(iw.word, index)

    return ans


# Link: https://leetcode.com/problems/word-abbreviation/description/
class IndexedWord:
  def __init__(self, word: str, index: int):
    self.word = word
    self.index = index


class Solution:
  def wordsAbbreviation(self, words: List[str]) -> List[str]:
    n = len(words)
    ans = [''] * n

    def getAbbrev(s: str, prefixIndex: int) -> str:
      n = len(s)
      num = n - (prefixIndex + 1) - 1
      numLength = 1 if num < 10 else (2 if num < 100 else 3)
      abbrevLength = (prefixIndex + 1) + numLength + 1
      if abbrevLength >= n:
        return s
      return s[:prefixIndex + 1] + str(num) + s[-1]

    abbrevToIndexedWords = collections.defaultdict(list)

    for i, word in enumerate(words):
      abbrev = getAbbrev(word, 0)
      abbrevToIndexedWords[abbrev].append(IndexedWord(word, i))

    def longestCommonPrefix(s1: str, s2: str) -> int:
      i = 0
      while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
      return i

    for indexedWords in abbrevToIndexedWords.values():
      indexedWords.sort(key=lambda x: x.word)
      lcp = [0] * len(indexedWords)
      for i, (a, b) in enumerate(zip(indexedWords, indexedWords[1:])):
        k = longestCommonPrefix(a.word, b.word)
        lcp[i] = max(lcp[i], k)
        lcp[i + 1] = k
      for iw, l in zip(indexedWords, lcp):
        ans[iw.index] = getAbbrev(iw.word, l)

    return ans


# Link: https://leetcode.com/problems/word-abbreviation/description/
class Solution:
  def wordsAbbreviation(self, words: List[str]) -> List[str]:
    n = len(words)

    def getAbbrev(s: str, prefixIndex: int) -> str:
      n = len(s)
      num = n - (prefixIndex + 1) - 1
      numLength = 1 if num < 10 else (2 if num < 100 else 3)
      abbrevLength = (prefixIndex + 1) + numLength + 1
      if abbrevLength >= n:
        return s
      return s[:prefixIndex + 1] + str(num) + s[-1]

    ans = [getAbbrev(word, 0) for word in words]
    # prefix[i] := ans[i] takes words[i][0..prefix[i]]
    prefix = [0] * n

    for i in range(n):
      while True:
        dupeIndices = []
        for j in range(i + 1, n):
          if ans[i] == ans[j]:
            dupeIndices.append(j)
        if not dupeIndices:
          break
        dupeIndices.append(i)
        for index in dupeIndices:
          prefix[index] += 1
          ans[index] = getAbbrev(words[index], prefix[index])

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/description/
from sortedcontainers import SortedList


class Solution:
  def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
    tasks.sort()
    workers.sort()

    def canComplete(k: int, pillsLeft: int) -> bool:
      """Returns True if we can finish k tasks."""
      # k strongest workers
      sortedWorkers = SortedList(workers[-k:])

      # Out of the k smallest tasks, start from the biggest one.
      for i in reversed(range(k)):
        # Find the first worker that has strength >= tasks[i].
        index = sortedWorkers.bisect_left(tasks[i])
        if index < len(sortedWorkers):
          sortedWorkers.pop(index)
        elif pillsLeft > 0:
          # Find the first worker that has strength >= tasks[i] - strength.
          index = sortedWorkers.bisect_left(tasks[i] - strength)
          if index < len(sortedWorkers):
            sortedWorkers.pop(index)
            pillsLeft -= 1
          else:
            return False
        else:
          return False

      return True

    ans = 0
    l = 0
    r = min(len(tasks), len(workers))

    while l <= r:
      m = (l + r) // 2
      if canComplete(m, pills):
        ans = m
        l = m + 1
      else:
        r = m - 1

    return ans


# Link: https://leetcode.com/problems/escape-a-large-maze/description/
class Solution:
  def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
    def dfs(i: int, j: int, target: List[int], visited: set) -> bool:
      if not 0 <= i < 10**6 or not 0 <= j < 10**6 or (i, j) in blocked or (i, j) in visited:
        return False

      visited.add((i, j))
      if len(visited) > (1 + 199) * 199 // 2 or [i, j] == target:
        return True
      return dfs(i + 1, j, target, visited) or \
          dfs(i - 1, j, target, visited) or \
          dfs(i, j + 1, target, visited) or \
          dfs(i, j - 1, target, visited)

    blocked = set(tuple(b) for b in blocked)
    return dfs(source[0], source[1], target, set()) and dfs(target[0], target[1], source, set())


# Link: https://leetcode.com/problems/count-the-number-of-houses-at-a-certain-distance-ii/description/
class Solution:
  # Same as 3015. Count the Number of Houses at a Certain Distance I
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


# Link: https://leetcode.com/problems/sliding-window-maximum/description/
class Solution:
  def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    ans = []
    maxQ = collections.deque()

    for i, num in enumerate(nums):
      while maxQ and maxQ[-1] < num:
        maxQ.pop()
      maxQ.append(num)
      if i >= k and nums[i - k] == maxQ[0]:  # out-of-bounds
        maxQ.popleft()
      if i >= k - 1:
        ans.append(maxQ[0])

    return ans


# Link: https://leetcode.com/problems/find-maximum-non-decreasing-array-length/description/
class Solution:
  def findMaximumLength(self, nums: List[int]) -> int:
    n = len(nums)
    # prefix[i] := the sum of the first i nums
    prefix = list(itertools.accumulate(nums, initial=0))
    # dp[i] := the maximum number of elements in the increasing
    # sequence after processing the first i nums
    dp = [0] * (n + 1)
    # bestLeft[i] := the index l s.t. merging nums[l..i) is the
    # optimal strategy among processing the first i nums
    bestLeft = [0] * (n + 2)

    for i in range(1, n + 1):
      bestLeft[i] = max(bestLeft[i], bestLeft[i - 1])
      # When merging nums[l, i), consider the next segment as [i, r).
      # Find the minimum `r` where sum(nums[l, i)) <= sum(nums[i, r)).
      # Equivalently, prefix[i] - prefix[l] <= prefix[r] - prefix[i].
      #            => prefix[r] >= prefix[i] * 2 - prefix[l]
      # Therefore, we can binary search `prefix` to find the minimum `r`.
      l = bestLeft[i]
      r = bisect.bisect_left(prefix, 2 * prefix[i] - prefix[l])
      dp[i] = dp[l] + 1
      bestLeft[r] = i

    return dp[n]


# Link: https://leetcode.com/problems/find-maximum-non-decreasing-array-length/description/
class Solution:
  def findMaximumLength(self, nums: List[int]) -> int:
    n = len(nums)
    kInf = 10_000_000_000
    # prefix[i] := the sum of the first i nums
    prefix = list(itertools.accumulate(nums, initial=0))
    # dp[i] := the maximum number of elements in the increasing
    # sequence after processing the first i nums
    dp = [0] * (n + 1)
    # last[i] := the last sum after processing the first i nums
    last = [0] + [kInf] * n

    for i in range(n):
      j = self._findIndex(i, prefix, last)
      dp[i + 1] = max(dp[i], dp[j] + 1)
      last[i + 1] = prefix[i + 1] - prefix[j]

    return dp[n]

  def _findIndex(self, i: int, prefix: List[int], last: List[int]) -> int:
    """Returns the index in [0..i].

    Returns the maximum index j in [0..i] s.t.
    prefix[i + 1] - prefix[j] >= last[j].
    """
    for j in range(i, -1, -1):
      if prefix[i + 1] - prefix[j] >= last[j]:
        return j


# Link: https://leetcode.com/problems/next-palindrome-using-same-digits/description/
class Solution:
  def nextPalindrome(self, num: str) -> str:
    def nextPermutation(nums: List[int]) -> bool:
      n = len(nums)

      # From the back to the front, find the first num < nums[i + 1].
      i = n - 2
      while i >= 0:
        if nums[i] < nums[i + 1]:
          break
        i -= 1

      if i < 0:
        return False

      # From the back to the front, find the first num > nums[i] and swap it
      # with nums[i].
      for j in range(n - 1, i, -1):
        if nums[j] > nums[i]:
          nums[i], nums[j] = nums[j], nums[i]
          break

      def reverse(nums, l, r):
        while l < r:
          nums[l], nums[r] = nums[r], nums[l]
          l += 1
          r -= 1

      # Reverse nums[i + 1..n - 1].
      reverse(nums, i + 1, len(nums) - 1)
      return True

    n = len(num)
    A = [ord(num[i]) - ord('0') for i in range(len(num) // 2)]

    if not nextPermutation(A):
      return ''

    s = ''.join([chr(ord('0') + a) for a in A])
    if n & 1:
      return s + num[n // 2] + s[::-1]
    return s + s[::-1]


# Link: https://leetcode.com/problems/palindrome-partitioning-ii/description/
class Solution:
  def minCut(self, s: str) -> int:
    n = len(s)
    # isPalindrome[i][j] := True if s[i..j] is a palindrome
    isPalindrome = [[True] * n for _ in range(n)]
    # dp[i] := the minimum cuts needed for a palindrome partitioning of s[0..i]
    dp = [n] * n

    for l in range(2, n + 1):
      i = 0
      for j in range(l - 1, n):
        isPalindrome[i][j] = s[i] == s[j] and isPalindrome[i + 1][j - 1]
        i += 1

    for i in range(n):
      if isPalindrome[0][i]:
        dp[i] = 0
        continue

      # Try all the possible partitions.
      for j in range(i):
        if isPalindrome[j + 1][i]:
          dp[i] = min(dp[i], dp[j] + 1)

    return dp[-1]


# Link: https://leetcode.com/problems/fancy-sequence/description/
class Fancy:
  def __init__(self):
    self.kMod = 1_000_000_007
    # For each `val` in `vals`, it actually represents a * val + b.
    self.vals = []
    self.a = 1
    self.b = 0

  # To undo a * val + b and get the original value, we append (val - b) // a.
  # By Fermat's little theorem:
  #   a^(p - 1) ≡ 1 (mod p)
  #   a^(p - 2) ≡ a^(-1) (mod p)
  # So, (val - b) / a ≡ (val - b) * a^(p - 2) (mod p)
  def append(self, val: int) -> None:
    x = (val - self.b + self.kMod) % self.kMod
    self.vals.append(x * pow(self.a, self.kMod - 2, self.kMod))

  # If the value is a * val + b, then the value after adding by `inc` will be
  # a * val + b + inc. So, we adjust b to b + inc.
  def addAll(self, inc: int) -> None:
    self.b = (self.b + inc) % self.kMod

  # If the value is a * val + b, then the value after multiplying by `m` will
  # be a * m * val + b * m. So, we adjust a to a * m and b to b * m.
  def multAll(self, m: int) -> None:
    self.a = (self.a * m) % self.kMod
    self.b = (self.b * m) % self.kMod

  def getIndex(self, idx: int) -> int:
    return -1 if idx >= len(self.vals) \
        else (self.a * self.vals[idx] + self.b) % self.kMod


# Link: https://leetcode.com/problems/online-majority-element-in-subarray/description/
class MajorityChecker:
  def __init__(self, arr: List[int]):
    self.arr = arr
    self.kTimes = 20  # 2^kTimes >> |arr|
    self.numToIndices = collections.defaultdict(list)

    for i, a in enumerate(self.arr):
      self.numToIndices[a].append(i)

  def query(self, left: int, right: int, threshold: int) -> int:
    for _ in range(self.kTimes):
      randIndex = random.randint(left, right)
      num = self.arr[randIndex]
      indices = self.numToIndices[num]
      l = bisect.bisect_left(indices, left)
      r = bisect.bisect_right(indices, right)
      if r - l >= threshold:
        return num

    return -1


# Link: https://leetcode.com/problems/length-of-the-longest-valid-substring/description/
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

  def search(self, word: str, l: int, r: int) -> bool:
    node: TrieNode = self.root
    for i in range(l, r):
      if word[i] not in node.children:
        return False
      node = node.children[word[i]]
    return node.isWord


class Solution:
  def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    ans = 0
    trie = Trie()

    for s in forbidden:
      trie.insert(s)

    # r is the rightmost index to make word[l..r] a valid substring.
    r = len(word) - 1
    for l in range(len(word) - 1, -1, -1):
      for end in range(l, min(l + 10, r + 1)):
        if trie.search(word, l, end + 1):
          r = end - 1
          break
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/length-of-the-longest-valid-substring/description/
class Solution:
  def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    forbiddenSet = set(forbidden)
    ans = 0
    r = len(word) - 1  # rightmost index of the valid substring

    for l in range(len(word) - 1, -1, -1):
      for end in range(l, min(l + 10, r + 1)):
        if word[l:end + 1] in forbiddenSet:
          r = end - 1
          break
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/number-of-unique-good-subsequences/description/
class Solution:
  # Similar to 940. Distinct Subsequences II
  def numberOfUniqueGoodSubsequences(self, binary: str) -> int:
    kMod = 1_000_000_007
    # endsIn[i] := the number of subsequence that end in ('0' + i)
    endsIn = {'0': 0, '1': 0}

    for c in binary:
      endsIn[c] = sum(endsIn.values()) % kMod
      # Don't count '0' since we want to avoid the leading zeros case.
      # However, we can always count '1'.
      if c == '1':
        endsIn['1'] += 1

    # Count '0' in the end.
    return (sum(endsIn.values()) + ('0' in binary)) % kMod


# Link: https://leetcode.com/problems/count-of-sub-multisets-with-bounded-sum/description/
class Solution:
  def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of submultisets of `nums` with sum i
    dp = [1] + [0] * r
    count = collections.Counter(nums)
    zeros = count.pop(0, 0)

    for num, freq in count.items():
      # stride[i] := dp[i] + dp[i - num] + dp[i - 2 * num] + ...
      stride = dp.copy()
      for i in range(num, r + 1):
        stride[i] += stride[i - num]
      for i in range(r, 0, -1):
        if i >= num * (freq + 1):
          # dp[i] + dp[i - num] + dp[i - freq * num]
          dp[i] = stride[i] - stride[i - num * (freq + 1)]
        else:
          dp[i] = stride[i]

    return (zeros + 1) * sum(dp[l:r + 1]) % kMod


# Link: https://leetcode.com/problems/super-palindromes/description/
class Solution:
  def superpalindromesInRange(self, left: str, right: str) -> int:
    def nextPalindrome(num: int) -> int:
      s = str(num)
      n = len(s)

      half = s[0:(n + 1) // 2]
      reversedHalf = half[:n // 2][::-1]
      candidate = int(half + reversedHalf)
      if candidate >= num:
        return candidate

      half = str(int(half) + 1)
      reversedHalf = half[:n // 2][::-1]
      return int(half + reversedHalf)

    def isPalindrome(num: int) -> bool:
      s = str(num)
      l = 0
      r = len(s) - 1

      while l < r:
        if s[l] != s[r]:
          return False
        l += 1
        r -= 1

      return True

    ans = 0
    l = int(left)
    r = int(right)
    i = int(sqrt(l))

    while i * i <= r:
      palindrome = nextPalindrome(i)
      squared = palindrome**2
      if squared <= r and isPalindrome(squared):
        ans += 1
      i = palindrome + 1

    return ans


# Link: https://leetcode.com/problems/russian-doll-envelopes/description/
class Solution:
  def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    # Same as 300. Longest Increasing Subsequence
    ans = 0
    dp = [0] * len(envelopes)

    for _, h in envelopes:
      l = 0
      r = ans
      while l < r:
        m = (l + r) // 2
        if dp[m] >= h:
          r = m
        else:
          l = m + 1
      dp[l] = h
      if l == ans:
        ans += 1

    return ans


# Link: https://leetcode.com/problems/subtree-removal-game-with-fibonacci-tree/description/
class Solution:
  def findGameWinner(self, n: int) -> bool:
    return n % 6 != 1


# Link: https://leetcode.com/problems/find-substring-with-given-hash-value/description/
class Solution:
  def subStrHash(self, s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
    maxPower = pow(power, k, modulo)
    hash = 0

    def val(c: str) -> int:
      return ord(c) - ord('a') + 1

    for i, c in reversed(list(enumerate(s))):
      hash = (hash * power + val(c)) % modulo
      if i + k < len(s):
        hash = (hash - val(s[i + k]) * maxPower) % modulo
      if hash == hashValue:
        bestLeft = i

    return s[bestLeft:bestLeft + k]


# Link: https://leetcode.com/problems/maximum-number-of-robots-within-budget/description/
class Solution:
  def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
    cost = 0
    maxQ = collections.deque()  # Stores `chargeTimes[i]`.

    j = 0  # window's range := [i..j], so k = i - j + 1
    for i, (chargeTime, runningCost) in enumerate(zip(chargeTimes, runningCosts)):
      cost += runningCost
      while maxQ and maxQ[-1] < chargeTime:
        maxQ.pop()
      maxQ.append(chargeTime)
      if maxQ[0] + (i - j + 1) * cost > budget:
        if maxQ[0] == chargeTimes[j]:
          maxQ.popleft()
        cost -= runningCosts[j]
        j += 1

    return len(chargeTimes) - j


# Link: https://leetcode.com/problems/reaching-points/description/
class Solution:
  def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
    while sx < tx and sy < ty:
      tx, ty = tx % ty, ty % tx

    return sx == tx and sy <= ty and (ty - sy) % tx == 0 or \
        sy == ty and sx <= tx and (tx - sx) % ty == 0


# Link: https://leetcode.com/problems/next-greater-element-iv/description/
class Solution:
  def secondGreaterElement(self, nums: List[int]) -> List[int]:
    ans = [-1] * len(nums)
    # a decreasing stack that stores indices that met the first greater number.
    prevStack = []
    # a decreasing stack that stores indices.
    currStack = []

    for i, num in enumerate(nums):
      # Indices in prevStack meet the second greater num.
      while prevStack and nums[prevStack[-1]] < num:
        ans[prevStack.pop()] = num
      # Push indices that meet the first greater number from `currStack` to
      # `prevStack`. We need a temporary array to make the indices in the
      # `prevStack` increasing.
      decreasingIndices = []
      while currStack and nums[currStack[-1]] < num:
        decreasingIndices.append(currStack.pop())
      while decreasingIndices:
        prevStack.append(decreasingIndices.pop())
      currStack.append(i)

    return ans


# Link: https://leetcode.com/problems/minimum-reverse-operations/description/
from sortedcontainers import SortedList


class Solution:
  def minReverseOperations(self, n: int, p: int, banned: List[int], k: int) -> List[int]:
    bannedSet = set(banned)
    ans = [-1] * n
    # unseen[i] := the unseen numbers that % 2 == i
    unseen = [SortedList(), SortedList()]

    for num in range(n):
      if num != p and num not in bannedSet:
        unseen[num & 1].add(num)

    # Perform BFS from `p`.
    q = collections.deque([p])
    ans[p] = 0

    while q:
      u = q.popleft()
      lo = max(u - k + 1, k - 1 - u)
      hi = min(u + k - 1, n - 1 - (u - (n - k)))
      # Choose the correct set of numbers.
      nums = unseen[lo & 1]
      i = nums.bisect_left(lo)
      while i < len(nums) and nums[i] <= hi:
        num = nums[i]
        ans[num] = ans[u] + 1
        q.append(num)
        nums.pop(i)

    return ans


# Link: https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/description/
class Solution:
  def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
    kMod = 1_000_000_007
    m = len(grid)
    n = len(grid[0])

    @functools.lru_cache(None)
    def dp(i: int, j: int, summ: int) -> int:
      """
      Returns the number of paths to (i, j), where the sum / k == `summ`.
      """
      if i == m or j == n:
        return 0
      if i == m - 1 and j == n - 1:
        return 1 if (summ + grid[i][j]) % k == 0 else 0
      newSum = (summ + grid[i][j]) % k
      return (dp(i + 1, j, newSum) + dp(i, j + 1, newSum)) % kMod

    return dp(0, 0, 0)


# Link: https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/description/
class Solution:
  def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
    kMod = 1_000_000_007
    m = len(grid)
    n = len(grid[0])
    # dp[i][j][sum] := the number of paths to (i, j), where the sum / k == sum
    dp = [[[0] * k for j in range(n)] for i in range(m)]
    dp[0][0][grid[0][0] % k] = 1

    for i in range(m):
      for j in range(n):
        for summ in range(k):
          newSum = (summ + grid[i][j]) % k
          if i > 0:
            dp[i][j][newSum] += dp[i - 1][j][summ]
          if j > 0:
            dp[i][j][newSum] += dp[i][j - 1][summ]
          dp[i][j][newSum] %= kMod

    return dp[m - 1][n - 1][0]


# Link: https://leetcode.com/problems/reconstruct-itinerary/description/
class Solution:
  def findItinerary(self, tickets: List[List[str]]) -> List[str]:
    ans = []
    graph = collections.defaultdict(list)

    for a, b in reversed(sorted(tickets)):
      graph[a].append(b)

    def dfs(u: str) -> None:
      while u in graph and graph[u]:
        dfs(graph[u].pop())
      ans.append(u)

    dfs('JFK')
    return ans[::-1]


# Link: https://leetcode.com/problems/make-the-xor-of-all-segments-equal-to-zero/description/
class Solution:
  def minChanges(self, nums: List[int], k: int) -> int:
    kMax = 1024
    n = len(nums)
    # counts[i] := the counter that maps at the i-th position
    counts = [collections.Counter() for _ in range(k)]
    # dp[i][j] := the minimum number of elements to change s.t. XOR(nums[i..k - 1]) is j
    dp = [[n] * kMax for _ in range(k)]

    for i, num in enumerate(nums):
      counts[i % k][num] += 1

    def countAt(i: int) -> int:
      return n // k + (1 if n % k > i else 0)

    # Initialize the DP array.
    for j in range(kMax):
      dp[k - 1][j] = countAt(k - 1) - counts[k - 1][j]

    for i in range(k - 2, -1, -1):
      # The worst-case scenario is changing all the i-th position numbers to a
      # non-existent value in the current bucket.
      changeAll = countAt(i) + min(dp[i + 1])
      for j in range(kMax):
        dp[i][j] = changeAll
        for num, freq in counts[i].items():
          # the cost to change every number in the i-th position to `num`
          cost = countAt(i) - freq
          dp[i][j] = min(dp[i][j], dp[i + 1][j ^ num] + cost)

    return dp[0][0]


# Link: https://leetcode.com/problems/maximum-building-height/description/
class Solution:
  def maxBuilding(self, n: int, restrictions: List[List[int]]) -> int:
    A = sorted(restrictions + [[1, 0]] + [[n, n - 1]])

    for i in range(len(A)):
      dist = A[i][0] - A[i - 1][0]
      A[i][1] = min(A[i][1], A[i - 1][1] + dist)

    for i in reversed(range(len(A) - 1)):
      dist = A[i + 1][0] - A[i][0]
      A[i][1] = min(A[i][1], A[i + 1][1] + dist)

    ans = 0

    for (l, hL), (r, hR) in zip(A, A[1:]):
      ans = max(ans, max(hL, hR) + (r - l - abs(hL - hR)) // 2)

    return ans


# Link: https://leetcode.com/problems/find-array-given-subset-sums/description/
class Solution:
  def recoverArray(self, n: int, sums: List[int]) -> List[int]:
    def recover(sums: List[int]) -> List[int]:
      if len(sums) == 1:
        return []

      count = collections.Counter(sums)
      # Either num or -num must be in the final array.
      #  num + sumsExcludingNum = sumsIncludingNum
      # -num + sumsIncludingNum = sumsExcludingNum
      num = sums[1] - sums[0]
      sumsExcludingNum = []
      sumsIncludingNum = []
      chooseSumsExcludingNum = True

      for summ in sums:
        if count[summ] == 0:
          continue
        count[summ] -= 1
        count[summ + num] -= 1
        sumsExcludingNum.append(summ)
        sumsIncludingNum.append(summ + num)
        if summ + num == 0:
          chooseSumsExcludingNum = False

      # Choose `sumsExludingNum` by default since we want to gradually strip
      # `num` from each sum in `sums` to have the final array. However, we should
      # always choose the group of sums with 0 since it's a must-have.
      return [num] + recover(sumsExcludingNum) if chooseSumsExcludingNum \
          else [-num] + recover(sumsIncludingNum)

    return recover(sorted(sums))


# Link: https://leetcode.com/problems/student-attendance-record-ii/description/
class Solution:
  def checkRecord(self, n: int) -> int:
    kMod = 1_000_000_007
    # dp[i][j] := the length so far with i A's and the last letters are j L's
    dp = [[0] * 3 for _ in range(2)]
    dp[0][0] = 1

    for _ in range(n):
      prev = [A[:] for A in dp]

      # Append a P.
      dp[0][0] = (prev[0][0] + prev[0][1] + prev[0][2]) % kMod

      # Append an L.
      dp[0][1] = prev[0][0]

      # Append an L.
      dp[0][2] = prev[0][1]

      # Append an A or append a P.
      dp[1][0] = (prev[0][0] + prev[0][1] + prev[0][2] +
                  prev[1][0] + prev[1][1] + prev[1][2]) % kMod

      # Append an L.
      dp[1][1] = prev[1][0]

      # Append an L.
      dp[1][2] = prev[1][1]

    return (sum(dp[0]) + sum(dp[1])) % kMod


# Link: https://leetcode.com/problems/merge-k-sorted-lists/description/
from queue import PriorityQueue


class Solution:
  def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    pq = PriorityQueue()

    for i, lst in enumerate(lists):
      if lst:
        pq.put((lst.val, i, lst))

    while not pq.empty():
      _, i, minNode = pq.get()
      if minNode.next:
        pq.put((minNode.next.val, i, minNode.next))
      curr.next = minNode
      curr = curr.next

    return dummy.next


# Link: https://leetcode.com/problems/find-beautiful-indices-in-the-given-array-ii/description/
class Solution:
  # Same as 3006. Find Beautiful Indices in the Given Array I
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

    lps = getLPS(pattern)
    res = []
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


# Link: https://leetcode.com/problems/choose-numbers-from-two-arrays-in-range/description/
class Solution:
  def countSubranges(self, nums1: List[int], nums2: List[int]) -> int:
    kMod = 1_000_000_007
    ans = 0
    # {sum, count}, add if choose from nums1, minus if choose from nums2
    dp = collections.Counter()

    for a, b in zip(nums1, nums2):
      newDp = collections.Counter()
      newDp[a] += 1
      newDp[-b] += 1

      for prevSum, count in dp.items():
        # Choose nums1[i]
        newDp[prevSum + a] += count
        newDp[prevSum + a] %= kMod
        # Choose nums2[i]
        newDp[prevSum - b] += count
        newDp[prevSum - b] %= kMod

      dp = newDp
      ans += dp[0]
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/word-search-ii/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.word: Optional[str] = None


class Solution:
  def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    m = len(board)
    n = len(board[0])
    ans = []
    root = TrieNode()

    def insert(word: str) -> None:
      node = root
      for c in word:
        node = node.children.setdefault(c, TrieNode())
      node.word = word

    for word in words:
      insert(word)

    def dfs(i: int, j: int, node: TrieNode) -> None:
      if i < 0 or i == m or j < 0 or j == n:
        return
      if board[i][j] == '*':
        return

      c = board[i][j]
      if c not in node.children:
        return

      child = node.children[c]
      if child.word:
        ans.append(child.word)
        child.word = None

      board[i][j] = '*'
      dfs(i + 1, j, child)
      dfs(i - 1, j, child)
      dfs(i, j + 1, child)
      dfs(i, j - 1, child)
      board[i][j] = c

    for i in range(m):
      for j in range(n):
        dfs(i, j, root)

    return ans


# Link: https://leetcode.com/problems/number-of-valid-move-combinations-on-chessboard/description/
class Solution:
  def countCombinations(self, pieces: List[str], positions: List[List[int]]) -> int:
    n = len(pieces)
    moves = {"rook": [(1, 0), (-1, 0), (0, 1), (0, -1)],
             "bishop": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
             "queen": [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]}
    ans = set()

    def getHash(board: List[List[int]]) -> Tuple:
      return tuple([tuple(pos) for pos in board])

    def dfs(board: List[List[int]], combMove: List[Tuple[int, int]], activeMask: int) -> None:
      if activeMask == 0:
        return
      ans.add(getHash(board))

      for nextActiveMask in range(1, 1 << n):
        if activeMask & nextActiveMask != nextActiveMask:
          continue

        # Make sure to copy the board.
        nextBoard = [pos.copy() for pos in board]

        # Move the pieces that are active in this turn.
        for i in range(n):
          if (nextActiveMask >> i) & 1:
            nextBoard[i][0] += combMove[i][0]
            nextBoard[i][1] += combMove[i][1]

        # No two or more pieces occupy the same square.
        if len(set(getHash(nextBoard))) < n:
          continue

        # Everything needs to be in the boundary.
        if all(1 <= x <= 8 and 1 <= y <= 8 for x, y in nextBoard):
          dfs(nextBoard, combMove, nextActiveMask)

    for combMove in product(*(moves[piece] for piece in pieces)):
      dfs(positions, combMove, (1 << n) - 1)

    return len(ans)


# Link: https://leetcode.com/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/description/
class Solution:
  def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
    kMax = 26
    m = int(math.log2(n)) + 1
    ans = []
    graph = [[] for _ in range(n)]
    # jump[i][j] := the node you reach after jumping 2^j from i
    jump = [[0] * m for _ in range(n)]
    # count[i][j] := the count of j from root to i, where 1 <= j <= 26
    count = [[] for _ in range(n)]
    # depth[i] := the depth of i
    depth = [0] * n

    for u, v, w in edges:
      graph[u].append((v, w))
      graph[v].append((u, w))

    def dfs(u: int, prev: int, d: int):
      if prev != -1:
        jump[u][0] = prev
      depth[u] = d
      for v, w in graph[u]:
        if v == prev:
          continue
        # Inherit the count from the parent.
        count[v] = count[u][:]
        # Add one to this edge.
        count[v][w] += 1
        dfs(v, u, d + 1)

    count[0] = [0] * (kMax + 1)
    dfs(0, -1, 0)

    # Calculate binary lifting.
    for j in range(1, m):
      for i in range(n):
        jump[i][j] = jump[jump[i][j - 1]][j - 1]

    def getLCA(u: int, v: int) -> int:
      """Returns the lca(u, v) via Calculate binary lifting."""
      # v is always deeper than u.
      if depth[u] > depth[v]:
        return getLCA(v, u)
      # Jump v to the same height of u.
      for j in range(m):
        if depth[v] - depth[u] >> j & 1:
          v = jump[v][j]
      if u == v:
        return u
      # Jump u and v to the node right below the lca.
      for j in range(m - 1, -1, -1):
        if jump[u][j] != jump[v][j]:
          u = jump[u][j]
          v = jump[v][j]
      return jump[v][0]

    for u, v in queries:
      lca = getLCA(u, v)
      # the number of edges between (u, v).
      numEdges = depth[u] + depth[v] - 2 * depth[lca]
      # the maximum frequency of edges between (u, v)
      maxFreq = max(count[u][j] + count[v][j] - 2 * count[lca][j]
                    for j in range(1, kMax + 1))
      ans.append(numEdges - maxFreq)

    return ans


# Link: https://leetcode.com/problems/car-fleet-ii/description/
class Solution:
  def getCollisionTimes(self, cars: List[List[int]]) -> List[float]:
    ans = []
    stack = []  # (pos, speed, collisionTime)

    def getCollisionTime(
            car: Tuple[int, int, int],
            pos: int, speed: int) -> float:
      return (car[0] - pos) / (speed - car[1])

    for pos, speed in reversed(cars):
      while stack and (
              speed <= stack[-1][1] or getCollisionTime(stack[-1],
                                                        pos, speed) >=
              stack[-1][2]):
        stack.pop()
      if stack:
        collisionTime = getCollisionTime(stack[-1], pos, speed)
        stack.append((pos, speed, collisionTime))
        ans.append(collisionTime)
      else:
        stack.append((pos, speed, math.inf))
        ans.append(-1)

    return ans[::-1]


# Link: https://leetcode.com/problems/palindrome-pairs/description/
class Solution:
  def palindromePairs(self, words: List[str]) -> List[List[int]]:
    ans = []
    dict = {word[::-1]: i for i, word in enumerate(words)}

    for i, word in enumerate(words):
      if "" in dict and dict[""] != i and word == word[::-1]:
        ans.append([i, dict[""]])

      for j in range(1, len(word) + 1):
        l = word[:j]
        r = word[j:]
        if l in dict and dict[l] != i and r == r[::-1]:
          ans.append([i, dict[l]])
        if r in dict and dict[r] != i and l == l[::-1]:
          ans.append([dict[r], i])

    return ans


# Link: https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/description/
class Solution:
  def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
    ans = []
    # tail[i] := the minimum tail of all the increasing subsequences having
    # length i + 1
    # It's easy to see that tail must be an increasing array
    tail = []

    for obstacle in obstacles:
      if not tail or obstacle >= tail[-1]:
        tail.append(obstacle)
        ans.append(len(tail))
      else:
        index = bisect.bisect_right(tail, obstacle)
        tail[index] = obstacle
        ans.append(index + 1)

    return ans


# Link: https://leetcode.com/problems/number-of-ways-to-reach-destination-in-the-grid/description/
class Solution:
  def numberOfWays(self, n: int, m: int, k: int, source: List[int], dest: List[int]) -> int:
    kMod = 1_000_000_007
    # the number of ways of `source` to `dest` using steps so far
    ans = int(source == dest)
    # the number of ways of `source` to dest's row using steps so far
    row = int(source[0] == dest[0] and source[1] != dest[1])
    # the number of ways of `source` to dest's col using steps so far
    col = int(source[0] != dest[0] and source[1] == dest[1])
    # the number of ways of `source` to others using steps so far
    others = int(source[0] != dest[0] and source[1] != dest[1])

    for _ in range(k):
      nextAns = (row + col) % kMod
      nextRow = (ans * (m - 1) +  # -self
                 row * (m - 2) +  # -self, -center
                 others) % kMod
      nextCol = (ans * (n - 1) +  # -self
                 col * (n - 2) +  # -self, -center
                 others) % kMod
      nextOthers = (row * (n - 1) +  # -self
                    col * (m - 1) +  # -self
                    others * (m + n - 1 - 3)) % kMod  # -self, -row, -col
      ans = nextAns
      row = nextRow
      col = nextCol
      others = nextOthers

    return ans


# Link: https://leetcode.com/problems/number-of-ways-to-reach-destination-in-the-grid/description/
class Solution:
  def numberOfWays(self, n: int, m: int, k: int, source: List[int], dest: List[int]) -> int:
    kMod = 1_000_000_007
    # dp[i][0] := the the number of ways of `source` to `dest` using i steps
    # dp[i][1] := the the number of ways of `source` to dest's row using i steps
    # dp[i][2] := the the number of ways of `source` to dest's col using i steps
    # dp[i][3] := the the number of ways of `source` to others using i steps
    dp = [[0] * 4 for _ in range(k + 1)]
    if source == dest:
      dp[0][0] = 1
    elif source[0] == dest[0]:
      dp[0][1] = 1
    elif source[1] == dest[1]:
      dp[0][2] = 1
    else:
      dp[0][3] = 1

    for i in range(1, k + 1):
      dp[i][0] = (dp[i - 1][1] + dp[i - 1][2]) % kMod
      dp[i][1] = (dp[i - 1][0] * (m - 1) +  # -self
                  dp[i - 1][1] * (m - 2) +  # -self, -center
                  dp[i - 1][3]) % kMod
      dp[i][2] = (dp[i - 1][0] * (n - 1) +  # -self
                  dp[i - 1][2] * (n - 2) +  # -self, -center
                  dp[i - 1][3]) % kMod
      dp[i][3] = (dp[i - 1][1] * (n - 1) +  # -self
                  dp[i - 1][2] * (m - 1) +  # -self
                  dp[i - 1][3] * (m + n - 1 - 3)) % kMod  # -self, -row, -col

    return dp[k][0]


# Link: https://leetcode.com/problems/sum-of-special-evenly-spaced-elements-in-array/description/
class Solution:
  def solve(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    kMod = 10**9 + 7
    n = len(nums)
    sqrtN = int(n**0.5)
    # prefix[x][y] = sum(nums[x + ay]), where a >= 0 and x + ay < n
    # Set prefix[i][j] to nums[i] to indicate the sequence starts with nums[i].
    prefix = [[num] * sqrtN for num in nums]

    for x in range(n - 1, -1, -1):
      for y in range(1, sqrtN):
        if x + y < n:
          prefix[x][y] += prefix[x + y][y]
          prefix[x][y] %= kMod

    return [prefix[x][y] if y < sqrtN
            else sum(nums[x::y]) % kMod
            for x, y in queries]


# Link: https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/description/
class Solution:
  def minOperations(self, target: List[int], arr: List[int]) -> int:
    indices = []
    numToIndex = {num: i for i, num in enumerate(target)}

    for a in arr:
      if a in numToIndex:
        indices.append(numToIndex[a])

    return len(target) - self._lengthOfLIS(indices)

  # Same as 300. Longest Increasing Subsequence
  def _lengthOfLIS(self, nums: List[int]) -> int:
    # tail[i] := the minimum tail of all the increasing subsequences having
    # length i + 1
    # It's easy to see that tail must be an increasing array.
    tail = []

    for num in nums:
      if not tail or num > tail[-1]:
        tail.append(num)
      else:
        tail[bisect.bisect_left(tail, num)] = num

    return len(tail)


# Link: https://leetcode.com/problems/equal-rational-numbers/description/
class Solution:
  def isRationalEqual(self, s: str, t: str) -> bool:
    ratios = [1, 1 / 9, 1 / 99, 1 / 999, 1 / 9999]

    def valueOf(s: str) -> float:
      if s.find('(') == -1:
        return float(s)

      # Get the indices.
      leftParenIndex = s.find('(')
      rightParenIndex = s.find(')')
      dotIndex = s.find('.')

      # integerAndNonRepeating := <IntegerPart><.><NonRepeatingPart>
      integerAndNonRepeating = float(s[:leftParenIndex])
      nonRepeatingLength = leftParenIndex - dotIndex - 1

      # repeating := <RepeatingPart>
      repeating = int(s[leftParenIndex + 1:rightParenIndex])
      repeatingLength = rightParenIndex - leftParenIndex - 1
      return integerAndNonRepeating + repeating * 0.1**nonRepeatingLength * ratios[repeatingLength]

    return abs(valueOf(s) - valueOf(t)) < 1e-9


# Link: https://leetcode.com/problems/maximize-the-number-of-partitions-after-operations/description/
class Solution:
  def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
    @functools.lru_cache(None)
    def dp(i: int, canChange: bool, mask: int) -> int:
      """
      Returns the maximum number of partitions of s[i..n), where `canChange` is
      True if we can still change a letter, and `mask` is the bitmask of the
      letters we've seen.
      """
      if i == len(s):
        return 0

      def getRes(newBit: int, nextCanChange: bool) -> int:
        nextMask = mask | newBit
        if nextMask.bit_count() > k:
          return 1 + dp(i + 1, nextCanChange, newBit)
        return dp(i + 1, nextCanChange, nextMask)

      # Initialize the result based on the current letter.
      res = getRes(1 << (ord(s[i]) - ord('a')), canChange)

      # If allowed, explore the option to change the current letter.
      if canChange:
        for j in range(26):
          res = max(res, getRes(1 << j, False))
      return res

    return dp(0, True, 0) + 1


# Link: https://leetcode.com/problems/parallel-courses-ii/description/
class Solution:
  def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
    # dp[i] := the minimum number of semesters to take the courses, where i is
    # the bitmask of the taken courses
    dp = [n] * (1 << n)
    # prereq[i] := bitmask of all dependencies of course i
    prereq = [0] * n

    for prevCourse, nextCourse in relations:
      prereq[nextCourse - 1] |= 1 << prevCourse - 1

    dp[0] = 0  # Don't need time to finish 0 course.

    for i in range(1 << n):
      # the bitmask of all the courses can be taken
      coursesCanBeTaken = 0
      # Can take the j-th course if i contains all of j's prerequisites.
      for j in range(n):
        if (i & prereq[j]) == prereq[j]:
          coursesCanBeTaken |= 1 << j
      # Don't take any course which is already taken.
      # (i represents set of courses that are already taken)
      coursesCanBeTaken &= ~i
      # Enumerate every bitmask subset of `coursesCanBeTaken`.
      s = coursesCanBeTaken
      while s:
        if bin(s).count('1') <= k:
          # Any combination of courses (if <= k) can be taken now.
          # i | s := combining courses taken with courses can be taken.
          dp[i | s] = min(dp[i | s], dp[i] + 1)
        s = (s - 1) & coursesCanBeTaken

    return dp[-1]


# Link: https://leetcode.com/problems/cat-and-mouse/description/
from enum import IntEnum


class State(IntEnum):
  kDraw = 0
  kMouseWin = 1
  kCatWin = 2


class Solution:
  def catMouseGame(self, graph: List[List[int]]) -> int:
    n = len(graph)
    # result of (cat, mouse, move)
    # move := 0 (mouse) // 1 (cat)
    states = [[[0] * 2 for i in range(n)] for j in range(n)]
    outDegree = [[[0] * 2 for i in range(n)] for j in range(n)]
    q = collections.deque()  # (cat, mouse, move, state)

    for cat in range(n):
      for mouse in range(n):
        outDegree[cat][mouse][0] = len(graph[mouse])
        outDegree[cat][mouse][1] = len(graph[cat]) - graph[cat].count(0)

    # Start from the states s.t. the winner can be determined.
    for cat in range(1, n):
      for move in range(2):
        # Mouse is in the hole.
        states[cat][0][move] = int(State.kMouseWin)
        q.append((cat, 0, move, int(State.kMouseWin)))
        # Cat catches mouse.
        states[cat][cat][move] = int(State.kCatWin)
        q.append((cat, cat, move, int(State.kCatWin)))

    while q:
      cat, mouse, move, state = q.popleft()
      if cat == 2 and mouse == 1 and move == 0:
        return state
      prevMove = move ^ 1
      for prev in graph[cat if prevMove else mouse]:
        prevCat = prev if prevMove else cat
        if prevCat == 0:  # invalid
          continue
        prevMouse = mouse if prevMove else prev
        # The state has been determined.
        if states[prevCat][prevMouse][prevMove]:
          continue
        if prevMove == 0 and state == int(State.kMouseWin) or \
                prevMove == 1 and state == int(State.kCatWin):
          states[prevCat][prevMouse][prevMove] = state
          q.append((prevCat, prevMouse, prevMove, state))
        else:
          outDegree[prevCat][prevMouse][prevMove] -= 1
          if outDegree[prevCat][prevMouse][prevMove] == 0:
            states[prevCat][prevMouse][prevMove] = state
            q.append((prevCat, prevMouse, prevMove, state))

    return states[2][1][0]


# Link: https://leetcode.com/problems/minimum-edge-reversals-so-every-node-is-reachable/description/
class Solution:
  def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append((v, True))  # 1 means (u -> v)
      graph[v].append((u, False))  # 0 means (v <- u)

    seen = {0}

    @functools.lru_cache(None)
    def dp(u: int) -> int:
      """
      Returns the minimum number of edge reversals so node u can reach every
      node in its subtree.
      """
      res = 0
      for v, isForward in graph[u]:
        if v in seen:
          continue
        seen.add(v)
        res += dp(v) + (0 if isForward else 1)
      return res

    ans = [0] * n
    ans[0] = dp(0)

    def dfs(u: int) -> None:
      for v, isForward in graph[u]:
        if v in seen:
          continue
        seen.add(v)
        ans[v] = ans[u] + (1 if isForward else -1)
        dfs(v)

    seen = {0}
    dfs(0)
    return ans


# Link: https://leetcode.com/problems/count-increasing-quadruplets/description/
class Solution:
  def countQuadruplets(self, nums: List[int]) -> int:
    ans = 0
    # dp[j] := the number of triplets (i, j, k) where i < j < k and nums[i] < nums[k] <
    # nums[j]. Keep this information for l to use later.
    dp = [0] * len(nums)

    # k can be treated as l.
    for k in range(2, len(nums)):
      numLessThanK = 0
      # j can be treated as i.
      for j in range(k):
        if nums[j] < nums[k]:
          numLessThanK += 1  # nums[i] < nums[k]
          # nums[j] < nums[l], so we should add dp[j] since we find a new
          # quadruplets for (i, j, k, l).
          ans += dp[j]
        elif nums[j] > nums[k]:
          dp[j] += numLessThanK

    return ans


# Link: https://leetcode.com/problems/find-k-th-smallest-pair-distance/description/
class Solution:
  def smallestDistancePair(self, nums: List[int], k: int) -> int:
    nums.sort()

    def numPairDistancesNoGreaterThan(m: int) -> int:
      count = 0
      j = 1
      # For each index i, find the first index j s.t. nums[j] > nums[i] + m,
      # so numPairDistancesNoGreaterThan for the index i will be j - i - 1.
      for i, num in enumerate(nums):
        while j < len(nums) and nums[j] <= num + m:
          j += 1
        count += j - i - 1
      return count

    return bisect.bisect_left(
        range(0, nums[-1] - nums[0]), k,
        key=lambda m: numPairDistancesNoGreaterThan(m))


# Link: https://leetcode.com/problems/maximize-value-of-function-in-a-ball-passing-game/description/
class Solution:
  def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
    n = len(receiver)
    m = int(math.log2(k)) + 1
    ans = 0
    # jump[i][j] := the the node you reach after jumping 2^j steps from i
    jump = [[0] * m for _ in range(n)]
    # summ[i][j] := the sum of the first 2^j nodes you reach when jumping from i
    summ = [[0] * m for _ in range(n)]

    for i in range(n):
      jump[i][0] = receiver[i]
      summ[i][0] = receiver[i]

    # Calculate binary lifting.
    for j in range(1, m):
      for i in range(n):
        midNode = jump[i][j - 1]
        #   the the node you reach after jumping 2^j steps from i
        # = the node you reach after jumping 2^(j - 1) steps from i
        # + the node you reach after jumping another 2^(j - 1) steps
        jump[i][j] = jump[midNode][j - 1]
        #   the sum of the first 2^j nodes you reach when jumping from i
        # = the sum of the first 2^(j - 1) nodes you reach when jumping from i
        # + the sum of another 2^(j - 1) nodes you reach
        summ[i][j] = summ[i][j - 1] + summ[midNode][j - 1]

    for i in range(n):
      currSum = i
      currPos = i
      for j in range(m):
        if (k >> j) & 1 == 1:
          currSum += summ[currPos][j]
          currPos = jump[currPos][j]
      ans = max(ans, currSum)

    return ans


# Link: https://leetcode.com/problems/scramble-string/description/
class Solution:
  @functools.lru_cache(None)
  def isScramble(self, s1: str, s2: str) -> bool:
    if s1 == s2:
      return True
    if collections.Counter(s1) != collections.Counter(s2):
      return False

    for i in range(1, len(s1)):
      if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
        return True
      if self.isScramble(s1[:i], s2[len(s2) - i:]) and self.isScramble(s1[i:], s2[:len(s2) - i]):
        return True

    return False


# Link: https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/description/
class Solution:
  def smallestRange(self, nums: List[List[int]]) -> List[int]:
    minHeap = [(row[0], i, 0) for i, row in enumerate(nums)]
    heapq.heapify(minHeap)

    maxRange = max(row[0] for row in nums)
    minRange = heapq.nsmallest(1, minHeap)[0][0]
    ans = [minRange, maxRange]

    while len(minHeap) == len(nums):
      num, r, c = heapq.heappop(minHeap)
      if c + 1 < len(nums[r]):
        heapq.heappush(minHeap, (nums[r][c + 1], r, c + 1))
        maxRange = max(maxRange, nums[r][c + 1])
        minRange = heapq.nsmallest(1, minHeap)[0][0]
        if maxRange - minRange < ans[1] - ans[0]:
          ans[0], ans[1] = minRange, maxRange

    return ans


# Link: https://leetcode.com/problems/minimum-initial-energy-to-finish-tasks/description/
class Solution:
  def minimumEffort(self, tasks: List[List[int]]) -> int:
    ans = 0
    prevSaved = 0

    for actual, minimum in sorted(tasks, key=lambda x: x[0] - x[1]):
      if prevSaved < minimum:
        ans += minimum - prevSaved
        prevSaved = minimum - actual
      else:
        prevSaved -= actual

    return ans


# Link: https://leetcode.com/problems/closest-node-to-path-in-tree/description/
class Solution:
  def closestNode(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
    ans = []
    tree = [[] for _ in range(n)]
    dist = [[-1] * n for _ in range(n)]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def fillDist(start: int, u: int, d: int) -> None:
      dist[start][u] = d
      for v in tree[u]:
        if dist[start][v] == -1:
          fillDist(start, v, d + 1)

    for i in range(n):
      fillDist(i, i, 0)

    def findClosest(u: int, end: int, node: int, ans: int) -> int:
      for v in tree[u]:
        if dist[v][end] < dist[u][end]:
          return findClosest(v, end, node, ans if dist[ans][node] < dist[v][node] else v)
      return ans

    return [findClosest(start, end, node, start)
            for start, end, node in query]


# Link: https://leetcode.com/problems/difference-between-maximum-and-minimum-price-sum/description/
class Solution:
  def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
    ans = 0
    tree = [[] for _ in range(n)]
    maxSums = [0] * n  # maxSums[i] := the maximum the sum of path rooted at i

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def maxSum(u: int, prev: int) -> int:
      maxChildSum = 0
      for v in tree[u]:
        if prev != v:
          maxChildSum = max(maxChildSum, maxSum(v, u))
      maxSums[u] = price[u] + maxChildSum
      return maxSums[u]

    # Precalculate `maxSums`.
    maxSum(0, -1)

    def reroot(u: int, prev: int, parentSum: int) -> None:
      nonlocal ans
      # Get the top two subtree sums and the top one node index.
      maxSubtreeSum1 = 0
      maxSubtreeSum2 = 0
      maxNode = -1
      for v in tree[u]:
        if v == prev:
          continue
        if maxSums[v] > maxSubtreeSum1:
          maxSubtreeSum2 = maxSubtreeSum1
          maxSubtreeSum1 = maxSums[v]
          maxNode = v
        elif maxSums[v] > maxSubtreeSum2:
          maxSubtreeSum2 = maxSums[v]

      if len(tree[u]) == 1:
        ans = max(ans, parentSum, maxSubtreeSum1)

      for v in tree[u]:
        if v == prev:
          continue
        nextParentSum = \
            price[u] + max(parentSum, maxSubtreeSum2) if v == maxNode else \
            price[u] + max(parentSum, maxSubtreeSum1)
        reroot(v, u, nextParentSum)

    reroot(0, -1, 0)
    return ans


# Link: https://leetcode.com/problems/the-number-of-good-subsets/description/
class Solution:
  def numberOfGoodSubsets(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    n = 1 << len(primes)
    # dp[i] := the number of good subsets with set of primes = i bit mask
    dp = [1] + [0] * (n - 1)
    count = collections.Counter(nums)

    for num, freq in count.items():
      if num == 1:
        continue
      if any(num % squared == 0 for squared in [4, 9, 25]):
        continue
      numPrimesMask = sum(1 << i
                          for i, prime in enumerate(primes)
                          if num % prime == 0)
      for primesMask in range(n):
        # Skip since there're commen set of primes (becomes invalid subset)
        if primesMask & numPrimesMask > 0:
          continue
        nextPrimesMask = numPrimesMask | primesMask
        dp[nextPrimesMask] += dp[primesMask] * freq
        dp[nextPrimesMask] %= kMod

    return (1 << count[1]) * sum(dp[1:]) % kMod


# Link: https://leetcode.com/problems/minimum-incompatibility/description/
class Solution:
  def __init__(self):
    self.kMaxNum = 16

  def minimumIncompatibility(self, nums: List[int], k: int) -> int:
    kMaxCompatibility = (16 - 1) * (16 // 2)
    n = len(nums)
    subsetSize = n // k
    maxMask = 1 << n
    incompatibilities = self._getIncompatibilities(nums, subsetSize)

    # dp[i] := the minimum possible sum of incompatibilities of the subset
    # of numbers represented by the bitmask i
    dp = [kMaxCompatibility] * maxMask
    dp[0] = 0

    for mask in range(1, maxMask):
      # The number of 1s in `mask` isn't a multiple of `subsetSize`.
      if mask.bit_count() % subsetSize != 0:
        continue
      # https://cp-algorithms.com/algebra/all-submasks.html
      submask = mask
      while submask > 0:
        if incompatibilities[submask] != -1:  # valid submask
          dp[mask] = min(dp[mask], dp[mask - submask] +
                         incompatibilities[submask])
        submask = (submask - 1) & mask

    return dp[-1] if dp[-1] != kMaxCompatibility else -1

  def _getIncompatibilities(self, nums: List[int], subsetSize: int) -> List[int]:
    """
    Returns an incompatibilities array where
    * incompatibilities[i] := the incompatibility of the subset of numbers
      represented by the bitmask i
    * incompatibilities[i] := -1 if the number of 1s in the bitmask i is not
      `subsetSize`
    """
    maxMask = 1 << len(nums)
    incompatibilities = [-1] * maxMask
    for mask in range(maxMask):
      if mask.bit_count() == subsetSize and self._isUnique(nums, mask, subsetSize):
        incompatibilities[mask] = self._getIncompatibility(nums, mask)
    return incompatibilities

  def _isUnique(self, nums: List[int], mask: int, subsetSize: int) -> bool:
    """Returns True if the numbers selected by `mask` are unique."""
    used = 0
    for i, num in enumerate(nums):
      if mask >> i & 1:
        used |= 1 << num
    return used.bit_count() == subsetSize

  def _getIncompatibility(self, nums: List[int], mask: int) -> int:
    """
    Returns the incompatibility of the selected numbers represented by the
    `mask`.
    """
    mini = self.kMaxNum
    maxi = 0
    for i, num in enumerate(nums):
      if mask >> i & 1:
        maxi = max(maxi, num)
        mini = min(mini, num)
    return maxi - mini


# Link: https://leetcode.com/problems/maximum-number-of-groups-with-increasing-length/description/
class Solution:
  def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
    ans = 1  # the next target length
    availableLimits = 0

    for usageLimit in sorted(usageLimits):
      availableLimits += usageLimit
      # Can create groups 1, 2, ..., ans.
      if availableLimits >= ans * (ans + 1) // 2:
        ans += 1

    return ans - 1


# Link: https://leetcode.com/problems/stamping-the-grid/description/
class Solution:
  def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
    m = len(grid)
    n = len(grid[0])
    # A[i][j] := the number of 1s in grid[0..i)[0..j)
    A = [[0] * (n + 1) for _ in range(m + 1)]
    # B[i][j] := the number of ways to stamp the submatrix in [0..i)[0..j)
    B = [[0] * (n + 1) for _ in range(m + 1)]
    # fit[i][j] := true if the stamps can fit with the right-bottom at (i, j)
    fit = [[False] * n for _ in range(m)]

    for i in range(m):
      for j in range(n):
        A[i + 1][j + 1] = A[i + 1][j] + A[i][j + 1] - A[i][j] + grid[i][j]
        if i + 1 >= stampHeight and j + 1 >= stampWidth:
          x = i - stampHeight + 1
          y = j - stampWidth + 1
          if A[i + 1][j + 1] - A[x][j + 1] - A[i + 1][y] + A[x][y] == 0:
            fit[i][j] = True

    for i in range(m):
      for j in range(n):
        B[i + 1][j + 1] = B[i + 1][j] + B[i][j + 1] - B[i][j] + fit[i][j]

    for i in range(m):
      for j in range(n):
        if not grid[i][j]:
          x = min(i + stampHeight, m)
          y = min(j + stampWidth, n)
          if B[x][y] - B[i][y] - B[x][j] + B[i][j] == 0:
            return False

    return True


# Link: https://leetcode.com/problems/longest-duplicate-substring/description/
class Solution:
  def longestDupSubstring(self, s: str) -> str:
    kMod = 1_000_000_007
    bestStart = -1
    l = 1
    r = len(s)

    def val(c: str) -> int:
      return ord(c) - ord('a')

    # k := the length of the substring to be hashed
    def getStart(k: int) -> Optional[int]:
      maxPow = pow(26, k - 1, kMod)
      hashToStart = collections.defaultdict(list)
      h = 0

      # Compute the hash value of s[:k].
      for i in range(k):
        h = (h * 26 + val(s[i])) % kMod
      hashToStart[h].append(0)

      # Compute the rolling hash by Rabin Karp.
      for i in range(k, len(s)):
        startIndex = i - k + 1
        h = (h - maxPow * val(s[i - k])) % kMod
        h = (h * 26 + val(s[i])) % kMod
        if h in hashToStart:
          currSub = s[startIndex:startIndex + k]
          for start in hashToStart[h]:
            if s[start:start + k] == currSub:
              return startIndex
        hashToStart[h].append(startIndex)

    while l < r:
      m = (l + r) // 2
      start: Optional[int] = getStart(m)
      if start:
        bestStart = start
        l = m + 1
      else:
        r = m

    if bestStart == -1:
      return ''
    if getStart(l):
      return s[bestStart:bestStart + l]
    return s[bestStart:bestStart + l - 1]


# Link: https://leetcode.com/problems/maximum-score-of-spliced-array/description/
class Solution:
  def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
    def kadane(nums1: List[int], nums2: List[int]) -> int:
      """
      Returns the maximum gain of swapping some numbers in `nums1` with some
      numbers in `nums2`.
      """
      gain = 0
      maxGain = 0

      for num1, num2 in zip(nums1, nums2):
        gain = max(0, gain + num2 - num1)
        maxGain = max(maxGain, gain)

      return maxGain + sum(nums1)

    return max(kadane(nums1, nums2), kadane(nums2, nums1))


# Link: https://leetcode.com/problems/count-the-number-of-good-partitions/description/
class Solution:
  def numberOfGoodPartitions(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    ans = 1
    # lastSeen[num] := the index of the last time `num` appeared
    lastSeen = {}

    for i, num in enumerate(nums):
      lastSeen[num] = i

    # Track the maximum right index of each running partition by ensuring that
    # the first and last occurrences of a number fall within the same partition.
    maxRight = 0
    for i, num in enumerate(nums):
      if i > maxRight:
        # Start a new partition that starts from nums[i].
        # Each partition doubles the total number of good partitions.
        ans = ans * 2 % kMod
      maxRight = max(maxRight, lastSeen[num])

    return ans


# Link: https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/description/
class Solution:
  def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
    graph = [[] for _ in range(n)]
    reversedGraph = [[] for _ in range(n)]

    for u, v, w in edges:
      graph[u].append((v, w))
      reversedGraph[v].append((u, w))

    def dijkstra(graph: List[List[Tuple[int, int]]], src: int) -> List[int]:
      dist = [math.inf] * n
      minHeap = [(0, src)]  # (d, u)
      while minHeap:
        d, u = heapq.heappop(minHeap)
        if dist[u] != math.inf:
          continue
        dist[u] = d
        for v, w in graph[u]:
          heapq.heappush(minHeap, (d + w, v))
      return dist

    fromSrc1 = dijkstra(graph, src1)
    fromSrc2 = dijkstra(graph, src2)
    fromDest = dijkstra(reversedGraph, dest)
    minWeight = min(a + b + c for a, b, c in zip(fromSrc1, fromSrc2, fromDest))
    return -1 if minWeight == math.inf else minWeight


# Link: https://leetcode.com/problems/reducing-dishes/description/
class Solution:
  def maxSatisfaction(self, satisfaction: List[int]) -> int:
    ans = 0
    sumSatisfaction = 0

    for s in sorted(satisfaction, reverse=True):
      sumSatisfaction += s
      if sumSatisfaction <= 0:
        return ans
      ans += sumSatisfaction

    return ans


# Link: https://leetcode.com/problems/longest-happy-prefix/description/
class Solution:
  def longestPrefix(self, s: str) -> str:
    kBase = 26
    kMod = 1_000_000_007
    n = len(s)
    maxLength = 0
    pow = 1
    prefixHash = 0  # the hash of s[0..i]
    suffixHash = 0  # the hash of s[j..n)

    def val(c: str) -> int:
      return ord(c) - ord('a')

    j = n - 1
    for i in range(n - 1):
      prefixHash = (prefixHash * kBase + val(s[i])) % kMod
      suffixHash = (val(s[j]) * pow + suffixHash) % kMod
      pow = pow * kBase % kMod
      if prefixHash == suffixHash:
        maxLength = i + 1
      j -= 1

    return s[:maxLength]


# Link: https://leetcode.com/problems/count-subarrays-with-median-k/description/
class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    kIndex = nums.index(k)
    ans = 0
    count = collections.Counter()

    balance = 0
    for i in range(kIndex, -1, -1):
      if nums[i] < k:
        balance -= 1
      elif nums[i] > k:
        balance += 1
      count[balance] += 1

    balance = 0
    for i in range(kIndex, len(nums)):
      if nums[i] < k:
        balance -= 1
      elif nums[i] > k:
        balance += 1
      # The subarray that has balance == 0 or 1 having median equal to k.
      # So, add count[0 - balance] and count[1 - balance] to `ans`.
      ans += count[-balance] + count[1 - balance]

    return ans


# Link: https://leetcode.com/problems/count-pairs-of-nodes/description/
class Solution:
  def countPairs(self, n: int, edges: List[List[int]], queries: List[int]) -> List[int]:
    ans = [0] * len(queries)

    # count[i] := the number of edges of node i
    count = [0] * (n + 1)

    # shared[i][j] := the number of edges incident to i or j, where i < j
    shared = [collections.Counter() for _ in range(n + 1)]

    for u, v in edges:
      count[u] += 1
      count[v] += 1
      shared[min(u, v)][max(u, v)] += 1

    sortedCount = sorted(count)

    for k, query in enumerate(queries):
      i = 1
      j = n
      while i < j:
        if sortedCount[i] + sortedCount[j] > query:
          # sortedCount[i] + sortedCount[j] > query
          # sortedCount[i + 1] + sortedCount[j] > query
          # ...
          # sortedCount[j - 1] + sortedCount[j] > query
          # So, there are (j - 1) - i + 1 = j - i pairs > query
          ans[k] += j - i
          j -= 1
        else:
          i += 1
      for i in range(1, n + 1):
        for j, sh in shared[i].items():
          if count[i] + count[j] > query and count[i] + count[j] - sh <= query:
            ans[k] -= 1

    return ans


# Link: https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/description/
class Solution:
  def tilingRectangle(self, n: int, m: int) -> int:
    @functools.lru_cache(None)
    def dp(heights: int) -> int:
      minHeight = min(heights)
      if minHeight == n:  # All filled.
        return 0

      ans = m * n
      heightsList = list(heights)
      start = heightsList.index(minHeight)

      # Try to put square of different size that doesn't exceed the width/height.
      for sz in range(1, min(m - start + 1, n - minHeight + 1)):
        # heights[start..start + sz) must has the same height.
        if heights[start + sz - 1] != minHeight:
          break
        # Put a square of size `sz` to cover heights[start..start + sz).
        heightsList[start:start + sz] = [minHeight + sz] * sz
        ans = min(ans, dp(tuple(heightsList)))

      return 1 + ans

    return dp(tuple([0] * m))


# Link: https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/
class Solution:
  def shortestSubarray(self, nums: List[int], k: int) -> int:
    n = len(nums)
    ans = n + 1
    dq = collections.deque()
    prefix = [0] + list(itertools.accumulate(nums))

    for i in range(n + 1):
      while dq and prefix[i] - prefix[dq[0]] >= k:
        ans = min(ans, i - dq.popleft())
      while dq and prefix[i] <= prefix[dq[-1]]:
        dq.pop()
      dq.append(i)

    return ans if ans <= n else -1


# Link: https://leetcode.com/problems/count-number-of-possible-root-nodes/description/
class Solution:
  def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
    ans = 0
    n = len(edges) + 1
    graph = [[] for _ in range(n)]
    guessGraph = [set() for _ in range(n)]
    parent = [0] * n

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    for u, v in guesses:
      guessGraph[u].add(v)

    def dfs(u: int, prev: int) -> None:
      parent[u] = prev
      for v in graph[u]:
        if v != prev:
          dfs(v, u)

    # Precalculate `parent`.
    dfs(0, -1)

    # Calculate `correctGuess` for tree rooted at 0.
    correctGuess = sum(i in guessGraph[parent[i]] for i in range(1, n))

    def reroot(u: int, prev: int, correctGuess: int) -> None:
      nonlocal ans
      if u != 0:
        # The tree is rooted at u, so a guess edge (u, prev) will match the new
        # `parent` relationship.
        if prev in guessGraph[u]:
          correctGuess += 1
        # A guess edge (prev, u) matching the old `parent` relationship will no
        # longer be True.
        if u in guessGraph[prev]:
          correctGuess -= 1
      if correctGuess >= k:
        ans += 1
      for v in graph[u]:
        if v != prev:
          reroot(v, u, correctGuess)

    reroot(0, -1, correctGuess)
    return ans


# Link: https://leetcode.com/problems/longest-common-subpath/description/
class Solution:
  def __init__(self):
    self.kMod = 8_417_508_174_513
    self.kBase = 165_131

  def longestCommonSubpath(self, n: int, paths: List[List[int]]) -> int:
    l = 0
    r = len(paths[0])

    while l < r:
      m = l + (r - l + 1) // 2
      if self._checkCommonSubpath(paths, m):
        l = m
      else:
        r = m - 1

    return l

  def _checkCommonSubpath(self, paths: List[List[int]], m: int) -> bool:
    """
    Returns True if there's a common subpath of length m for all the paths.
    """
    # Calculate the hash values for subpaths of length m for every path.
    hashSets = [self._rabinKarp(path, m) for path in paths]

    # Check if there is a common subpath of length m.
    for subpathHash in hashSets[0]:
      if all(subpathHash in hashSet for hashSet in hashSets):
        return True

    return False

  def _rabinKarp(self, path: List[int], m: int) -> Set[int]:
    """Returns the hash values for subpaths of length m in the path."""
    hashes = set()
    maxPower = 1
    hash = 0

    for i, num in enumerate(path):
      hash = (hash * self.kBase + num) % self.kMod
      if i >= m:
        hash = (hash - path[i - m] * maxPower %
                self.kMod + self.kMod) % self.kMod
      else:
        maxPower = maxPower * self.kBase % self.kMod
      if i >= m - 1:
        hashes.add(hash)

    return hashes


# Link: https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/description/
class Solution:
  def minimumOneBitOperations(self, n: int) -> int:
    # Observation: e.g. n = 2^2
    #        100 (2^2 needs 2^3 - 1 ops)
    # op1 -> 101
    # op2 -> 111
    # op1 -> 110
    # op2 -> 010 (2^1 needs 2^2 - 1 ops)
    # op1 -> 011
    # op2 -> 001 (2^0 needs 2^1 - 1 ops)
    # op1 -> 000
    #
    # So 2^k needs 2^(k + 1) - 1 ops. Note this is reversible, i.e., 0 -> 2^k
    # also takes 2^(k + 1) - 1 ops.

    # e.g. n = 1XXX, our first goal is to change 1XXX -> 1100.
    #   - If the second bit is 1, you only need to consider the cost of turning
    #     the last 2 bits to 0.
    #   - If the second bit is 0, you need to add up the cost of flipping the
    #     second bit from 0 to 1.
    # XOR determines the cost minimumOneBitOperations(1XXX^1100) accordingly.
    # Then, 1100 -> 0100 needs 1 op. Finally, 0100 -> 0 needs 2^3 - 1 ops.
    if n == 0:
      return 0
    # x is the largest 2^k <= n.
    # x | x >> 1 -> x >> 1 needs 1 op.
    #     x >> 1 -> 0      needs x = 2^k - 1 ops.
    x = 1 << n.bit_length() - 1
    return self.minimumOneBitOperations(n ^ (x | x >> 1)) + 1 + x - 1


# Link: https://leetcode.com/problems/maximum-equal-frequency/description/
class Solution:
  def maxEqualFreq(self, nums: List[int]) -> int:
    ans = 0
    maxFreq = 0
    count = collections.Counter()
    freq = collections.Counter()

    for i, num in enumerate(nums):
      freq[count[num]] -= 1
      count[num] += 1
      freq[count[num]] += 1
      maxFreq = max(maxFreq, count[num])
      if maxFreq == 1 or maxFreq * freq[maxFreq] == i or (maxFreq - 1) * (freq[maxFreq - 1] + 1) == i:
        ans = i + 1

    return ans


# Link: https://leetcode.com/problems/recover-the-original-array/description/
class Solution:
  def recoverArray(self, nums: List[int]) -> List[int]:
    nums = sorted(nums)

    def getArray(x: int, count: collections.Counter) -> List[int]:
      A = []
      for num in nums:
        if count[num] == 0:
          continue
        if count[num + x] == 0:
          return []
        count[num] -= 1
        count[num + x] -= 1
        A.append(num + x // 2)
      return A

    count = collections.Counter(nums)

    for i in range(1, len(nums)):
      x = nums[i] - nums[0]  # 2 * k
      if x <= 0 or x & 1:
        continue
      A = getArray(x, count.copy())
      if A:
        return A


# Link: https://leetcode.com/problems/longest-path-with-different-adjacent-characters/description/
class Solution:
  def longestPath(self, parent: List[int], s: str) -> int:
    n = len(parent)
    ans = 0
    graph = [[] for _ in range(n)]

    for i in range(1, n):
      graph[parent[i]].append(i)

    def longestPathDownFrom(u: int) -> int:
      nonlocal ans
      max1 = 0
      max2 = 0

      for v in graph[u]:
        res = longestPathDownFrom(v)
        if s[u] == s[v]:
          continue
        if res > max1:
          max2 = max1
          max1 = res
        elif res > max2:
          max2 = res

      ans = max(ans, 1 + max1 + max2)
      return 1 + max1

    longestPathDownFrom(0)
    return ans


# Link: https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/
class Solution:
  def findSubstring(self, s: str, words: List[str]) -> List[int]:
    if len(s) == 0 or words == []:
      return []

    k = len(words)
    n = len(words[0])
    ans = []
    count = collections.Counter(words)

    for i in range(len(s) - k * n + 1):
      seen = collections.defaultdict(int)
      j = 0
      while j < k:
        word = s[i + j * n: i + j * n + n]
        seen[word] += 1
        if seen[word] > count[word]:
          break
        j += 1
      if j == k:
        ans.append(i)

    return ans


# Link: https://leetcode.com/problems/closest-subsequence-sum/description/
class Solution:
  def minAbsDifference(self, nums: List[int], goal: int) -> int:
    n = len(nums) // 2
    ans = math.inf
    lSums = []
    rSums = []

    def dfs(A: List[int], i: int, path: int, sums: List[int]) -> None:
      if i == len(A):
        sums.append(path)
        return
      dfs(A, i + 1, path + A[i], sums)
      dfs(A, i + 1, path, sums)

    dfs(nums[:n], 0, 0, lSums)
    dfs(nums[n:], 0, 0, rSums)
    rSums.sort()

    for lSum in lSums:
      i = bisect_left(rSums, goal - lSum)
      if i < len(rSums):  # 2^n
        ans = min(ans, abs(goal - lSum - rSums[i]))
      if i > 0:
        ans = min(ans, abs(goal - lSum - rSums[i - 1]))

    return ans


# Link: https://leetcode.com/problems/check-if-there-is-a-valid-parentheses-string-path/description/
class Solution:
  def hasValidPath(self, grid: List[List[str]]) -> bool:
    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> bool:
      """
      Returns True if there's a path from grid[i][j] to grid[m - 1][n - 1],
      where the number of '(' - the number of ')' == k.
      """
      if i == len(grid) or j == len(grid[0]):
        return False
      k += 1 if grid[i][j] == '(' else -1
      if k < 0:
        return False
      if i == len(grid) - 1 and j == len(grid[0]) - 1:
        return k == 0
      return dp(i + 1, j, k) | dp(i, j + 1, k)

    return dp(0, 0, 0)


# Link: https://leetcode.com/problems/last-day-where-you-can-still-cross/description/
class Solution:
  def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def canWalk(day: int) -> bool:
      matrix = [[0] * col for _ in range(row)]
      for i in range(day):
        x, y = cells[i]
        matrix[x - 1][y - 1] = 1

      q = collections.deque()

      for j in range(col):
        if matrix[0][j] == 0:
          q.append((0, j))
          matrix[0][j] = 1

      while q:
        i, j = q.popleft()
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if x < 0 or x == row or y < 0 or y == col:
            continue
          if matrix[x][y] == 1:
            continue
          if x == row - 1:
            return True
          q.append((x, y))
          matrix[x][y] = 1

      return False

    ans = 0
    l = 1
    r = len(cells) - 1

    while l <= r:
      m = (l + r) // 2
      if canWalk(m):
        ans = m
        l = m + 1
      else:
        r = m - 1

    return ans


# Link: https://leetcode.com/problems/number-of-squareful-arrays/description/
class Solution:
  def numSquarefulPerms(self, nums: List[int]) -> int:
    ans = 0
    used = [False] * len(nums)

    def isSquare(num: int) -> bool:
      root = int(math.sqrt(num))
      return root * root == num

    def dfs(path: List[int]) -> None:
      nonlocal ans
      if len(path) > 1 and not isSquare(path[-1] + path[-2]):
        return
      if len(path) == len(nums):
        ans += 1
        return

      for i, a in enumerate(nums):
        if used[i]:
          continue
        if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
          continue
        used[i] = True
        dfs(path + [a])
        used[i] = False

    nums.sort()
    dfs([])
    return ans


# Link: https://leetcode.com/problems/number-of-atoms/description/
class Solution:
  def countOfAtoms(self, formula: str) -> str:
    def parse() -> dict:
      ans = collections.defaultdict(int)

      nonlocal i
      while i < n:
        if formula[i] == '(':
          i += 1
          for elem, freq in parse().items():
            ans[elem] += freq
        elif formula[i] == ')':
          i += 1
          numStart = i
          while i < n and formula[i].isdigit():
            i += 1
          factor = int(formula[numStart:i])
          for elem, freq in ans.items():
            ans[elem] *= factor
          return ans
        elif formula[i].isupper():
          elemStart = i
          i += 1
          while i < n and formula[i].islower():
            i += 1
          elem = formula[elemStart:i]
          numStart = i
          while i < n and formula[i].isdigit():
            i += 1
          num = 1 if i == numStart else int(
              formula[numStart:i])
          ans[elem] += num

      return ans

    n = len(formula)

    ans = ""
    i = 0
    count = parse()

    for elem in sorted(count.keys()):
      ans += elem
      if count[elem] > 1:
        ans += str(count[elem])

    return ans


# Link: https://leetcode.com/problems/maximum-gcd-sum-of-a-subarray/description/
class Solution:
  def maxGcdSum(self, nums: List[int], k: int) -> int:
    ans = 0
    # [(startIndex, gcd of subarray starting at startIndex)]
    startIndexAndGcds = []
    prefix = [0] + list(itertools.accumulate(nums))

    for i, num in enumerate(nums):
      nextStartIndexAndGcds = []
      for startIndex, gcd in startIndexAndGcds:
        nextGcd = math.gcd(gcd, nums[i])
        if not nextStartIndexAndGcds or \
                nextStartIndexAndGcds[-1][1] != nextGcd:  # Skip duplicates.
          nextStartIndexAndGcds.append((startIndex, nextGcd))
      startIndexAndGcds = nextStartIndexAndGcds
      startIndexAndGcds.append((i, nums[i]))
      for startIndex, gcd in startIndexAndGcds:
        if i - startIndex + 1 >= k:
          ans = max(ans, (prefix[i + 1] - prefix[startIndex]) * gcd)

    return ans


# Link: https://leetcode.com/problems/count-the-number-of-infection-sequences/description/
class Solution:
  def numberOfSequence(self, n: int, sick: List[int]) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def fact(i: int) -> int:
      return 1 if i <= 1 else i * fact(i - 1) % kMod

    @functools.lru_cache(None)
    def inv(i: int) -> int:
      return pow(i, kMod - 2, kMod)

    ans = fact(n - len(sick))  # the number of infected children
    prevSick = -1

    for i, s in enumerate(sick):
      # The segment [prevSick + 1, sick - 1] are the current non-infected
      # children.
      nonInfected = sick[i] - prevSick - 1
      prevSick = sick[i]
      if nonInfected == 0:
        continue
      ans *= inv(fact(nonInfected))
      ans %= kMod
      if i > 0:
        # There're two choices per second since the children at the two
        # endpoints can both be the infect candidates. So, there are
        # 2^[nonInfected - 1] ways to infect all children in the current
        # segment.
        ans *= pow(2, nonInfected - 1, kMod)

    nonInfected = n - sick[-1] - 1
    return ans * inv(fact(nonInfected)) % kMod


# Link: https://leetcode.com/problems/remove-9/description/
class Solution:
  def newInteger(self, n: int) -> int:
    ans = []
    while n:
      ans.append(str(n % 9))
      n //= 9
    return ''.join(reversed(ans))


# Link: https://leetcode.com/problems/sum-of-total-strength-of-wizards/description/
class Solution:
  def totalStrength(self, strength: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(strength)
    # left[i] := the next index on the left (if any)
    #            s.t. nums[left[i]] <= nums[i]
    left = [-1] * n
    # right[i] := the next index on the right (if any)
    #             s.t. nums[right[i]] < nums[i]
    right = [n] * n
    stack = []

    for i in reversed(range(n)):
      while stack and strength[stack[-1]] >= strength[i]:
        left[stack.pop()] = i
      stack.append(i)

    stack = []

    for i in range(n):
      while stack and strength[stack[-1]] > strength[i]:
        right[stack.pop()] = i
      stack.append(i)

    ans = 0
    prefixOfPrefix = list(itertools.accumulate(
        itertools.accumulate(strength), initial=0))

    # For each strength[i] as the minimum, calculate sum.
    for i, (l, r) in enumerate(zip(left, right)):
      leftSum = prefixOfPrefix[i] - prefixOfPrefix[max(0, l)]
      rightSum = prefixOfPrefix[r] - prefixOfPrefix[i]
      leftLen = i - l
      rightLen = r - i
      ans += strength[i] * (rightSum * leftLen - leftSum * rightLen) % kMod

    return ans % kMod


# Link: https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/description/
class Solution:
  def shortestPath(self, grid: List[List[int]], k: int) -> int:
    m = len(grid)
    n = len(grid[0])
    if m == 1 and n == 1:
      return 0

    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    steps = 0
    q = collections.deque([(0, 0, k)])
    seen = {(0, 0, k)}

    while q:
      steps += 1
      for _ in range(len(q)):
        i, j, eliminate = q.popleft()
        for l in range(4):
          x = i + dirs[l]
          y = j + dirs[l + 1]
          if x < 0 or x == m or y < 0 or y == n:
            continue
          if x == m - 1 and y == n - 1:
            return steps
          if grid[x][y] == 1 and eliminate == 0:
            continue
          newEliminate = eliminate - grid[x][y]
          if (x, y, newEliminate) in seen:
            continue
          q.append((x, y, newEliminate))
          seen.add((x, y, newEliminate))

    return -1


# Link: https://leetcode.com/problems/match-substring-after-replacement/description/
class Solution:
  def matchReplacement(self, s: str, sub: str, mappings: List[List[str]]) -> bool:
    isMapped = [[False] * 128 for _ in range(128)]

    for old, new in mappings:
      isMapped[ord(old)][ord(new)] = True

    for i in range(len(s)):
      if self._canTransform(s, i, sub, isMapped):
        return True

    return False

  def _canTransform(self, s: str, start: int, sub: str, isMapped: List[List[bool]]) -> bool:
    if start + len(sub) > len(s):
      return False

    for i in range(len(sub)):
      a = sub[i]
      b = s[start + i]
      if a != b and not isMapped[ord(a)][ord(b)]:
        return False

    return True


# Link: https://leetcode.com/problems/parse-lisp-expression/description/
class Solution:
  def evaluate(self, expression: str) -> int:
    def evaluate(e: str, prevScope: dict) -> int:
      if e[0].isdigit() or e[0] == '-':
        return int(e)
      if e in prevScope:
        return prevScope[e]

      scope = prevScope.copy()
      nextExpression = e[e.index(' ') + 1:-1]
      tokens = parse(nextExpression)

      if e[1] == 'm':  # 'mult'
        return evaluate(tokens[0], scope) * evaluate(tokens[1], scope)
      if e[1] == 'a':  # 'add'
        return evaluate(tokens[0], scope) + evaluate(tokens[1], scope)

      # 'let'
      for i in range(0, len(tokens) - 2, 2):
        scope[tokens[i]] = evaluate(tokens[i + 1], scope)

      return evaluate(tokens[-1], scope)

    def parse(e: str):
      tokens = []
      s = ''
      opened = 0

      for c in e:
        if c == '(':
          opened += 1
        elif c == ')':
          opened -= 1
        if opened == 0 and c == ' ':
          tokens.append(s)
          s = ''
        else:
          s += c

      if len(s) > 0:
        tokens.append(s)
      return tokens

    return evaluate(expression, {})


# Link: https://leetcode.com/problems/count-number-of-special-subsequences/description/
class Solution:
  def countSpecialSubsequences(self, nums: List[int]) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(i: int, prev: int) -> int:
      """
      Returns the number of increasing subsequences of the first i numbers,
      where the the previous number is j - 1.
      """
      if i == len(nums):
        return prev == 2

      res = 0

      # Don't include `nums[i]`.
      res += dp(i + 1, prev)

      # Include `nums[i]`.
      if nums[i] == prev:
        res += dp(i + 1, prev)
      if prev == -1 and nums[i] == 0:
        res += dp(i + 1, 0)
      if prev == 0 and nums[i] == 1:
        res += dp(i + 1, 1)
      if prev == 1 and nums[i] == 2:
        res += dp(i + 1, 2)

      res %= kMod
      return res

    return dp(0, -1)


# Link: https://leetcode.com/problems/count-number-of-special-subsequences/description/
class Solution:
  def countSpecialSubsequences(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    # dp[j] := the number of increasing subsequences of the numbers so far that
    # end in j
    dp = [0] * 3

    if nums[0] == 0:
      dp[0] = 1

    for i in range(1, n):
      if nums[i] == 0:
        dp[0] = dp[0] * 2 + 1
      elif nums[i] == 1:
        dp[1] = dp[1] * 2 + dp[0]
      else:  # nums[i] == 2
        dp[2] = dp[2] * 2 + dp[1]

      for ending in range(3):
        dp[ending] %= kMod

    return dp[2]


# Link: https://leetcode.com/problems/count-number-of-special-subsequences/description/
class Solution:
  def countSpecialSubsequences(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    # dp[i][j] := the number of increasing subsequences of the first i numbers
    # that end in j
    dp = [[0] * 3 for _ in range(n)]

    if nums[0] == 0:
      dp[0][0] = 1

    for i in range(1, n):
      for ending in range(3):
        dp[i][ending] = dp[i - 1][ending]

      if nums[i] == 0:
        # 1. The number of the previous subsequences that end in 0.
        # 2. Append a 0 to the previous subsequences that end in 0.
        # 3. Start a new subsequence from this 0.
        dp[i][0] = dp[i - 1][0] * 2 + 1
      elif nums[i] == 1:
        # 1. The number of the previous subsequences that end in 1.
        # 2. Append a 1 to the previous subsequences that end in 1.
        # 3. Append a 1 to the previous subsequences that end in 0.
        dp[i][1] = dp[i - 1][1] * 2 + dp[i - 1][0]
      else:  # nums[i] == 2
        # 1. The number of the previous subsequences that end in 2.
        # 2. Append a 2 to the previous subsequences that end in 2.
        # 3. Append a 2 to the previous subsequences that end in 1.
        dp[i][2] = dp[i - 1][2] * 2 + dp[i - 1][1]

      for ending in range(3):
        dp[i][ending] %= kMod

    return dp[-1][2]


# Link: https://leetcode.com/problems/stickers-to-spell-word/description/
class Solution:
  def minStickers(self, stickers: List[str], target: str) -> int:
    maxMask = 1 << len(target)
    # dp[i] := the minimum number of stickers to spell out i, where i is the
    # bit mask of target
    dp = [math.inf] * maxMask
    dp[0] = 0

    for mask in range(maxMask):
      if dp[mask] == math.inf:
        continue
      # Try to expand from `mask` by using each sticker.
      for sticker in stickers:
        superMask = mask
        for c in sticker:
          for i, t in enumerate(target):
            # Try to apply it on a missing letter.
            if c == t and not (superMask >> i & 1):
              superMask |= 1 << i
              break
        dp[superMask] = min(dp[superMask], dp[mask] + 1)

    return -1 if dp[-1] == math.inf else dp[-1]


# Link: https://leetcode.com/problems/median-of-two-sorted-arrays/description/
class Solution:
  def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    n1 = len(nums1)
    n2 = len(nums2)
    if n1 > n2:
      return self.findMedianSortedArrays(nums2, nums1)

    l = 0
    r = n1

    while l <= r:
      partition1 = (l + r) // 2
      partition2 = (n1 + n2 + 1) // 2 - partition1
      maxLeft1 = -2**31 if partition1 == 0 else nums1[partition1 - 1]
      maxLeft2 = -2**31 if partition2 == 0 else nums2[partition2 - 1]
      minRight1 = 2**31 - 1 if partition1 == n1 else nums1[partition1]
      minRight2 = 2**31 - 1 if partition2 == n2 else nums2[partition2]
      if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
        return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) * 0.5 if (n1 + n2) % 2 == 0 else max(maxLeft1, maxLeft2)
      elif maxLeft1 > minRight2:
        r = partition1 - 1
      else:
        l = partition1 + 1


# Link: https://leetcode.com/problems/number-of-beautiful-partitions/description/
class Solution:
  def beautifulPartitions(self, s: str, k: int, minLength: int) -> int:
    def isPrime(c: str) -> bool:
      return c in '2357'

    if not isPrime(s[0]) or isPrime(s[-1]):
      return 0

    kMod = 1_000_000_007

    @lru_cache(None)
    def dp(i: int, k: int) -> int:
      """
      Returns the number of beautiful partitions of s[i..n) with k bars (|)
      left.
      """
      if i <= len(s) and k == 0:
        return 1
      if i >= len(s):
        return 0

      # Don't split between s[i - 1] and s[i].
      ans = dp(i + 1, k) % kMod

      # Split between s[i - 1] and s[i].
      if isPrime(s[i]) and not isPrime(s[i - 1]):
        ans += dp(i + minLength, k - 1)

      return ans % kMod

    return dp(minLength, k - 1)


# Link: https://leetcode.com/problems/groups-of-strings/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    self.sz = [1] * n

  def unionBySize(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.sz[i] < self.sz[j]:
      self.sz[j] += self.sz[i]
      self.id[i] = j
    else:
      self.sz[i] += self.sz[j]
      self.id[j] = i
    self.count -= 1

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def groupStrings(self, words: List[str]) -> List[int]:
    uf = UnionFind(len(words))

    def getMask(s: str) -> int:
      mask = 0
      for c in s:
        mask |= 1 << ord(c) - ord('a')
      return mask

    def getAddedMasks(mask: int):
      for i in range(26):
        if not (mask >> i & 1):
          yield mask | 1 << i

    def getDeletedMasks(mask: int):
      for i in range(26):
        if mask >> i & 1:
          yield mask ^ 1 << i

    maskToIndex = {getMask(word): i for i, word in enumerate(words)}
    deletedMaskToIndex = {}

    for i, word in enumerate(words):
      mask = getMask(word)
      for m in getAddedMasks(mask):
        if m in maskToIndex:
          uf.unionBySize(i, maskToIndex[m])
      for m in getDeletedMasks(mask):
        if m in maskToIndex:
          uf.unionBySize(i, maskToIndex[m])
        if m in deletedMaskToIndex:
          uf.unionBySize(i, deletedMaskToIndex[m])
        else:
          deletedMaskToIndex[m] = i

    return [uf.count, max(uf.sz)]


# Link: https://leetcode.com/problems/super-egg-drop/description/
class Solution:
  def superEggDrop(self, k: int, n: int) -> int:
    moves = 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]

    while dp[moves][k] < n:
      moves += 1
      for eggs in range(1, k + 1):
        dp[moves][eggs] = dp[moves - 1][eggs - 1] + \
            dp[moves - 1][eggs] + 1

    return moves


# Link: https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/description/
class IndexedQuery:
  def __init__(self, queryIndex: int, a: int, b: int):
    self.queryIndex = queryIndex
    self.a = a  # Alice's index
    self.b = b  # Bob's index

  def __iter__(self):
    yield self.queryIndex
    yield self.a
    yield self.b


class Solution:
  # Similar to 2736. Maximum Sum Queries
  def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
    ans = [-1] * len(queries)
    # Store indices (heightsIndex) of heights with heights[heightsIndex] in
    # descending order.
    stack = []

    # Iterate through queries and heights simultaneously.
    heightsIndex = len(heights) - 1
    for queryIndex, a, b in sorted([IndexedQuery(i, min(a, b), max(a, b))
                                    for i, (a, b) in enumerate(queries)],
                                   key=lambda iq: -iq.b):
      if a == b or heights[a] < heights[b]:
        # 1. Alice and Bob are already in the same index (a == b) or
        # 2. Alice can jump from a -> b (heights[a] < heights[b]).
        ans[queryIndex] = b
      else:
        # Now, a < b and heights[a] >= heights[b].
        # Gradually add heights with an index > b to the monotonic stack.
        while heightsIndex > b:
          # heights[heightsIndex] is a better candidate, given that
          # heightsIndex is smaller than the indices in the stack and
          # heights[heightsIndex] is larger or equal to the heights mapped in
          # the stack.
          while stack and heights[stack[-1]] <= heights[heightsIndex]:
            stack.pop()
          stack.append(heightsIndex)
          heightsIndex -= 1
        # Binary search to find the smallest index j such that j > b and
        # heights[j] > heights[a], thereby ensuring heights[j] > heights[b].
        j = self._lastGreater(stack, a, heights)
        if j != -1:
          ans[queryIndex] = stack[j]

    return ans

  def _lastGreater(self, A: List[int], target: int, heights: List[int]):
    """
    Returns the last index i in A s.t. heights[A.get(i)] is > heights[target].
    """
    l = -1
    r = len(A) - 1
    while l < r:
      m = (l + r + 1) // 2
      if heights[A[m]] > heights[target]:
        l = m
      else:
        r = m - 1
    return l


# Link: https://leetcode.com/problems/design-graph-with-shortest-path-calculator/description/
class Graph:
  def __init__(self, n: int, edges: List[List[int]]):
    self.graph = [[] for _ in range(n)]
    for edge in edges:
      self.addEdge(edge)

  def addEdge(self, edge: List[int]):
    u, v, w = edge
    self.graph[u].append((v, w))

  def shortestPath(self, node1: int, node2: int) -> int:
    dist = [math.inf] * len(self.graph)

    dist[node1] = 0
    minHeap = [(dist[node1], node1)]  # (d, u)

    while minHeap:
      d, u = heapq.heappop(minHeap)
      if u == node2:
        return d
      for v, w in self.graph[u]:
        if d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (dist[v], v))

    return -1


# Link: https://leetcode.com/problems/selling-pieces-of-wood/description/
class Solution:
  def sellingWood(self, m: int, n: int, prices: List[List[int]]) -> int:
    # dp[i][j] := the maximum money of cutting i x j piece of wood
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for h, w, price in prices:
      dp[h][w] = price

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        for h in range(1, i // 2 + 1):
          dp[i][j] = max(dp[i][j], dp[h][j] + dp[i - h][j])
        for w in range(1, j // 2 + 1):
          dp[i][j] = max(dp[i][j], dp[i][w] + dp[i][j - w])

    return dp[m][n]


# Link: https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points/description/
class Solution:
  def connectTwoGroups(self, cost: List[List[int]]) -> int:
    # minCosts[j] := the minimum cost of connecting group2's point j
    minCosts = [min(col) for col in zip(*cost)]

    @functools.lru_cache(None)
    def dp(i: int, mask: int) -> int:
      """
      Returns the minimum cost to connect group1's points[i..n) with group2's
      points, where `mask` is the bitmask of the connected points in group2.
      """
      if i == len(cost):
        # All the points in group 1 are connected, so greedily assign the
        # minimum cost for the unconnected points of group2.
        return sum(minCost for j, minCost in enumerate(minCosts)
                   if (mask & 1 << j) == 0)
      return min(cost[i][j] + dp(i + 1, mask | 1 << j)
                 for j in range(len(cost[0])))

    return dp(0, 0)


# Link: https://leetcode.com/problems/maximum-genetic-difference-query/description/
class TrieNode:
  def __init__(self):
    self.children = [None] * 2
    self.count = 0


class Trie:
  def __init__(self):
    self.root = TrieNode()
    self.kHeight = 17

  def update(self, num: int, val: int) -> None:
    node = self.root
    for i in range(self.kHeight, -1, -1):
      bit = (num >> i) & 1
      if not node.children[bit]:
        node.children[bit] = TrieNode()
      node = node.children[bit]
      node.count += val

  def query(self, num: int) -> int:
    ans = 0
    node = self.root
    for i in range(self.kHeight, -1, -1):
      bit = (num >> i) & 1
      targetBit = bit ^ 1
      if node.children[targetBit] and node.children[targetBit].count > 0:
        ans += 1 << i
        node = node.children[targetBit]
      else:
        node = node.children[targetBit ^ 1]
    return ans


class Solution:
  def maxGeneticDifference(self, parents: List[int], queries: List[List[int]]) -> List[int]:
    n = len(parents)
    ans = [0] * len(queries)
    rootVal = -1
    tree = [[] for _ in range(n)]
    nodeToQueries = collections.defaultdict(list)  # {node: (index, val)}
    trie = Trie()

    for i, parent in enumerate(parents):
      if parent == -1:
        rootVal = i
      else:
        tree[parent].append(i)

    for i, (node, val) in enumerate(queries):
      nodeToQueries[node].append((i, val))

    def dfs(node: int) -> None:
      trie.update(node, 1)

      # Answer queries for node
      for i, val in nodeToQueries[node]:
        ans[i] = trie.query(val)

      for child in tree[node]:
        dfs(child)

      trie.update(node, -1)

    dfs(rootVal)
    return ans


# Link: https://leetcode.com/problems/minimize-maximum-value-in-a-grid/description/
class Solution:
  def minScore(self, grid: List[List[int]]) -> List[List[int]]:
    m = len(grid)
    n = len(grid[0])
    ans = [[0] * n for _ in range(m)]
    valAndIndices = []
    rows = [0] * m  # rows[i] := the maximum used number so far
    cols = [0] * n  # cols[j] := the maximum used number so far

    for i in range(m):
      for j in range(n):
        valAndIndices.append((grid[i][j], i, j))

    valAndIndices.sort()

    for _, i, j in valAndIndices:
      nextAvailable = max(rows[i], cols[j]) + 1
      ans[i][j] = nextAvailable
      rows[i] = nextAvailable
      cols[j] = nextAvailable

    return ans


# Link: https://leetcode.com/problems/max-value-of-equation/description/
class Solution:
  def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
    ans = -math.inf
    maxQ = collections.deque()  # (y - x, x)

    for x, y in points:
      # Remove the invalid points, xj - xi > k
      while maxQ and x - maxQ[0][1] > k:
        maxQ.popleft()
      if maxQ:
        ans = max(ans, x + y + maxQ[0][0])
      # Remove the points that contribute less value and have a bigger x.
      while maxQ and y - x >= maxQ[-1][0]:
        maxQ.pop()
      maxQ.append((y - x, x))

    return ans


# Link: https://leetcode.com/problems/max-value-of-equation/description/
class Solution:
  def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
    ans = -math.inf
    maxHeap = []  # (y - x, x)

    for x, y in points:
      while maxHeap and x + maxHeap[0][1] > k:
        heapq.heappop(maxHeap)
      if maxHeap:
        ans = max(ans, x + y - maxHeap[0][0])
      heapq.heappush(maxHeap, (x - y, -x))

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-change-the-final-value-of-expression/description/
class Solution:
  def minOperationsToFlip(self, expression: str) -> int:
    stack = []  # [(the expression, the cost to toggle the expression)]

    for e in expression:
      if e in '(&|':
        # These aren't expressions, so the cost is meaningless.
        stack.append((e, 0))
        continue
      if e == ')':
        lastPair = stack.pop()
        stack.pop()  # Pop '('.
      else:  # e == '0' or e == '1'
        # Store the '0' or '1'. The cost to change their values is just 1,
        # whether it's changing '0' to '1' or '1' to '0'.
        lastPair = (e, 1)
      if stack and stack[-1][0] in '&|':
        op = stack.pop()[0]
        a, costA = stack.pop()
        b, costB = lastPair
        # Determine the cost to toggle op(a, b).
        if op == '&':
          if a == '0' and b == '0':
            # Change '&' to '|' and a|b to '1'.
            lastPair = ('0', 1 + min(costA, costB))
          elif a == '0' and b == '1':
            # Change '&' to '|'.
            lastPair = ('0', 1)
          elif a == '1' and b == '0':
            # Change '&' to '|'.
            lastPair = ('0', 1)
          else:  # a == '1' and b == '1'
            # Change a|b to '0'.
            lastPair = ('1', min(costA, costB))
        else:  # op == '|'
          if a == '0' and b == '0':
            # Change a|b to '1'.
            lastPair = ('0', min(costA, costB))
          elif a == '0' and b == '1':
            # Change '|' to '&'.
            lastPair = ('1', 1)
          elif a == '1' and b == '0':
            # Change '|' to '&'.
            lastPair = ('1', 1)
          else:  # a == '1' and b == '1'
            # Change '|' to '&' and a|b to '0'.
            lastPair = ('1', 1 + min(costA, costB))
      stack.append(lastPair)

    return stack[-1][1]


# Link: https://leetcode.com/problems/process-restricted-friend-requests/description/
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
  def friendRequests(self, n: int, restrictions: List[List[int]], requests: List[List[int]]) -> List[bool]:
    ans = []
    uf = UnionFind(n)

    for u, v in requests:
      pu = uf.find(u)
      pv = uf.find(v)
      isValid = True
      if pu != pv:
        for x, y in restrictions:
          px = uf.find(x)
          py = uf.find(y)
          if (pu, pv) in [(px, py), (py, px)]:
            isValid = False
            break
      ans.append(isValid)
      if isValid:
        uf.unionByRank(pu, pv)

    return ans


# Link: https://leetcode.com/problems/stream-of-characters/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.isWord = False


class StreamChecker:
  def __init__(self, words: List[str]):
    self.root = TrieNode()
    self.letters = []

    for word in words:
      self._insert(word)

  def query(self, letter: str) -> bool:
    self.letters.append(letter)
    node = self.root
    for c in reversed(self.letters):
      if c not in node.children:
        return False
      node = node.children[c]
      if node.isWord:
        return True
    return False

  def _insert(self, word: str) -> None:
    node = self.root
    for c in reversed(word):
      node = node.children.setdefault(c, TrieNode())
    node.isWord = True


# Link: https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/description/
class Solution:
  def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
    ans = 0
    q = collections.deque()
    reachedClosedBoxes = [0] * len(status)

    def pushBoxesIfPossible(boxes: List[int]) -> None:
      for box in boxes:
        if status[box]:
          q.append(box)
        else:
          reachedClosedBoxes[box] = True

    pushBoxesIfPossible(initialBoxes)

    while q:
      currBox = q.popleft()

      # Add the candies.
      ans += candies[currBox]

      # Push `reachedClosedBoxes` by `key` obtained in this turn and change
      # their statuses.
      for key in keys[currBox]:
        if not status[key] and reachedClosedBoxes[key]:
          q.append(key)
        status[key] = 1  # boxes[key] is now open

      # Push the boxes contained in `currBox`.
      pushBoxesIfPossible(containedBoxes[currBox])

    return ans


# Link: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
class Codec:
  def serialize(self, root: 'TreeNode') -> str:
    """Encodes a tree to a single string."""
    if not root:
      return ''

    s = ''
    q = collections.deque([root])

    while q:
      node = q.popleft()
      if node:
        s += str(node.val) + ' '
        q.append(node.left)
        q.append(node.right)
      else:
        s += 'n '

    return s

  def deserialize(self, data: str) -> 'TreeNode':
    """Decodes your encoded data to tree."""
    if not data:
      return None

    vals = data.split()
    root = TreeNode(vals[0])
    q = collections.deque([root])

    for i in range(1, len(vals), 2):
      node = q.popleft()
      if vals[i] != 'n':
        node.left = TreeNode(vals[i])
        q.append(node.left)
      if vals[i + 1] != 'n':
        node.right = TreeNode(vals[i + 1])
        q.append(node.right)

    return root


# Link: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/
class Codec:
  def serialize(self, root: 'TreeNode') -> str:
    """Encodes a tree to a single string."""
    s = []

    def preorder(root: 'TreeNode') -> None:
      if not root:
        s.append('n')
        return

      s.append(str(root.val))
      preorder(root.left)
      preorder(root.right)

    preorder(root)
    return ' '.join(s)

  def deserialize(self, data: str) -> 'TreeNode':
    """Decodes your encoded data to tree."""
    q = collections.deque(data.split())

    def preorder() -> 'TreeNode':
      s = q.popleft()
      if s == 'n':
        return None

      root = TreeNode(s)
      root.left = preorder()
      root.right = preorder()
      return root

    return preorder()


# Link: https://leetcode.com/problems/design-skiplist/description/
class Node:
  def __init__(self, val=-1, next=None, down=None):
    self.val = val
    self.next = next
    self.down = down


class Skiplist:
  def __init__(self):
    self.dummy = Node()

  def search(self, target: int) -> bool:
    node = self.dummy
    while node:
      while node.next and node.next.val < target:
        node = node.next
      if node.next and node.next.val == target:
        return True
      # Move to the next level
      node = node.down
    return False

  def add(self, num: int) -> None:
    # Collect nodes that are before the insertion point.
    nodes = []
    node = self.dummy
    while node:
      while node.next and node.next.val < num:
        node = node.next
      nodes.append(node)
      # Move to the next level
      node = node.down

    shouldInsert = True
    down = None
    while shouldInsert and nodes:
      node = nodes.pop()
      node.next = Node(num, node.next, down)
      down = node.next
      shouldInsert = random.getrandbits(1) == 0

    # Create a topmost new level dummy that points to the existing dummy.
    if shouldInsert:
      self.dummy = Node(-1, None, self.dummy)

  def erase(self, num: int) -> bool:
    node = self.dummy
    found = False
    while node:
      while node.next and node.next.val < num:
        node = node.next
      if node.next and node.next.val == num:
        # Delete the node
        node.next = node.next.next
        found = True
      # Move to the next level
      node = node.down
    return found

  # Move to the node s.t. node.next.val >= target
  def _advance(self, node: Node, target: int) -> None:
    while node.next and node.next.val < target:
      node = node.next


# Link: https://leetcode.com/problems/maximize-grid-happiness/description/
class Solution:
  def getMaxGridHappiness(self, m: int, n: int, introvertsCount: int, extrovertsCount: int) -> int:
    def getPlacementCost(i: int, j: int, inMask: int, exMask: int, diff: int) -> int:
      """Calculates the cost based on left and up neighbors.

      The `diff` parameter represents the happiness change due to the current
      placed person in (i, j). We add `diff` each time we encounter a neighbor
      (left or up) who is already placed.

      Case 1: If the neighbor is an introvert, we subtract 30 from cost.
      Case 2: If the neighbor is an extrovert, we add 20 to from cost.
      """
      cost = 0
      if i > 0:
        if (1 << (n - 1)) & inMask:
          cost += diff - 30
        if (1 << (n - 1)) & exMask:
          cost += diff + 20
      if j > 0:
        if 1 & inMask:
          cost += diff - 30
        if 1 & exMask:
          cost += diff + 20
      return cost

    @functools.lru_cache(None)
    def dp(pos: int, inMask: int, exMask: int, inCount: int, exCount: int) -> int:
      # `inMask` is the placement of introvert people in the last n cells.
      # e.g. if we have m = 2, n = 3, i = 1, j = 1, then inMask = 0b101 means
      #
      # ? 1 0
      # 1 x ? (x := current position)
      i, j = divmod(pos, n)
      if i == m:
        return 0

      shiftedInMask = (inMask << 1) & ((1 << n) - 1)
      shiftedExMask = (exMask << 1) & ((1 << n) - 1)

      skip = dp(pos + 1, shiftedInMask, shiftedExMask, inCount, exCount)
      placeIntrovert = 120 + getPlacementCost(i, j, inMask, exMask, -30) + \
          dp(pos + 1, shiftedInMask + 1, shiftedExMask, inCount - 1, exCount) if inCount > 0 \
          else -math.inf
      placeExtrovert = 40 + getPlacementCost(i, j, inMask, exMask, 20) + \
          dp(pos + 1, shiftedInMask, shiftedExMask + 1, inCount, exCount - 1) if exCount > 0 \
          else -math.inf
      return max(skip, placeIntrovert, placeExtrovert)

    return dp(0, 0, 0, introvertsCount, extrovertsCount)


# Link: https://leetcode.com/problems/minimum-total-distance-traveled/description/
class Solution:
  def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
    robot.sort()
    factory.sort()

    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> int:
      """
      Returns the minimum distance to fix robot[i..n) with factory[j..n), where
      factory[j] already fixed k robots.
      """
      if i == len(robot):
        return 0
      if j == len(factory):
        return math.inf
      skipFactory = dp(i, j + 1, 0)
      position, limit = factory[j]
      useFactory = dp(i + 1, j, k + 1) + abs(robot[i] - position) \
          if limit > k else math.inf
      return min(skipFactory, useFactory)

    return dp(0, 0, 0)


# Link: https://leetcode.com/problems/color-the-triangle-red/description/
class Solution:
  def colorRed(self, n: int) -> List[List[int]]:
    ans = []
    tipSize = n % 4

    # The tip of the triangle is always painted red.
    if tipSize >= 1:
      ans.append([1, 1])

    # Pamost right and most left elements at the following rows.
    for i in range(2, tipSize + 1):
      ans.append([i, 1])
      ans.append([i, 2 * i - 1])

    # Pa4-row chunks.
    for i in range(tipSize + 1, n, 4):
      # Fill the first row of the chunk.
      ans.append([i, 1])
      # Fill the second row.
      for j in range(1, i + 1):
        ans.append([i + 1, 2 * j + 1])
      # Fill the third row.
      ans.append([i + 2, 2])
      # Fill the fourth row.
      for j in range(i + 2 + 1):
        ans.append([i + 3, 2 * j + 1])

    return ans


# Link: https://leetcode.com/problems/largest-color-value-in-a-directed-graph/description/
class Solution:
  def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
    n = len(colors)
    ans = 0
    processed = 0
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n
    q = collections.deque()
    count = [[0] * 26 for _ in range(n)]

    # Build the graph.
    for u, v in edges:
      graph[u].append(v)
      inDegrees[v] += 1

    # Vpology
    for i, degree in enumerate(inDegrees):
      if degree == 0:
        q.append(i)

    while q:
      u = q.popleft()
      processed += 1
      count[u][ord(colors[u]) - ord('a')] += 1
      ans = max(ans, count[u][ord(colors[u]) - ord('a')])
      for v in graph[u]:
        for i in range(26):
          count[v][i] = max(count[v][i], count[u][i])
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return ans if processed == n else -1


# Link: https://leetcode.com/problems/k-empty-slots/description/
class Solution:
  def kEmptySlots(self, bulbs: List[int], k: int) -> int:
    n = len(bulbs)
    ans = math.inf
    # day[i] := the day when bulbs[i] is turned on
    day = [0] * n

    for i, bulb in enumerate(bulbs):
      day[bulb - 1] = i + 1

    # Find a subarray of day[l..r], where its length is k + 2.
    # For each l < i < r, day[i] > max(day[l], day[r]).
    l = 0
    r = l + k + 1
    i = 1
    while r < n:
      if i == r:
        ans = min(ans, max(day[l], day[r]))
        l = i
        r = i + k + 1
      elif day[i] < max(day[l], day[r]):
        l = i
        r = i + k + 1
      i += 1

    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/
class Solution:
  def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
    i = 0

    def recoverFromPreorder(depth: int) -> Optional[TreeNode]:
      nonlocal i
      nDashes = 0
      while i + nDashes < len(traversal) and traversal[i + nDashes] == '-':
        nDashes += 1
      if nDashes != depth:
        return None

      i += depth
      start = i
      while i < len(traversal) and traversal[i].isdigit():
        i += 1

      return TreeNode(int(traversal[start:i]),
                      recoverFromPreorder(depth + 1),
                      recoverFromPreorder(depth + 1))

    return recoverFromPreorder(0)


# Link: https://leetcode.com/problems/minimum-flips-in-binary-tree-to-get-result/description/
class Solution:
  def minimumFlips(self, root: Optional[TreeNode], result: bool) -> int:
    @functools.lru_cache(None)
    def dp(root: Optional[TreeNode], target: bool) -> int:
      """Returns the minimum flips to make the subtree root become target."""
      if root.val in (0, 1):  # the leaf
        return 0 if root.val == target else 1
      if root.val == 5:  # NOT
        return dp(root.left or root.right, not target)
      if root.val == 2:  # OR
        nextTargets = [(0, 1), (1, 0), (1, 1)] if target else [[0, 0]]
      elif root.val == 3:  # AND
        nextTargets = [(1, 1)] if target else [(0, 0), (0, 1), (1, 0)]
      else:  # root.val == 4 XOR
        nextTargets = [(0, 1), (1, 0)] if target else [(0, 0), (1, 1)]
      return min(dp(root.left, leftTarget) + dp(root.right, rightTarget)
                 for leftTarget, rightTarget in nextTargets)

    return dp(root, result)


# Link: https://leetcode.com/problems/shortest-palindrome/description/
class Solution:
  def shortestPalindrome(self, s: str) -> str:
    t = s[::-1]

    for i in range(len(t)):
      if s.startswith(t[i:]):
        return t[:i] + s

    return t + s


# Link: https://leetcode.com/problems/self-crossing/description/
class Solution:
  def isSelfCrossing(self, x: List[int]) -> bool:
    if len(x) <= 3:
      return False

    for i in range(3, len(x)):
      if x[i - 2] <= x[i] and x[i - 1] <= x[i - 3]:
        return True
      if i >= 4 and x[i - 1] == x[i - 3] and x[i - 2] <= x[i] + x[i - 4]:
        return True
      if i >= 5 and x[i - 4] <= x[i - 2] and x[i - 2] <= x[i] + x[i - 4] and x[i - 1] <= x[i - 3] and x[i - 3] <= x[i - 1] + x[i - 5]:
        return True

    return False


# Link: https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string/description/
class Solution:
  def uniqueLetterString(self, s: str) -> int:
    ans = 0
    # lastSeen[c] := the index of the last time ('a' + i) appeared
    lastSeen = collections.defaultdict(lambda: -1)
    # prevSeen[c] := the previous index of the last time ('a' + i) appeared
    prevLastSeen = collections.defaultdict(lambda: -1)

    for i, c in enumerate(s):
      if c in lastSeen:
        ans += (i - lastSeen[c]) * (lastSeen[c] - prevLastSeen[c])
      prevLastSeen[c] = lastSeen[c]
      lastSeen[c] = i

    for c in string.ascii_uppercase:
      ans += (len(s) - lastSeen[c]) * (lastSeen[c] - prevLastSeen[c])

    return ans


# Link: https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string/description/
class Solution:
  def uniqueLetterString(self, s: str) -> int:
    ans = 0
    # the number of unique letters in all the substrings ending in the index so
    # far
    dp = 0
    lastCount = {}
    lastSeen = {}

    for i, c in enumerate(s):
      newCount = i - lastSeen.get(c, -1)
      # Substract the duplicates.
      dp -= lastCount.get(c, 0)
      # Add count of s[lastSeen[c] + 1..i], s[lastSeen[c] + 2..i], ..., s[i].
      dp += newCount
      lastCount[c] = newCount
      lastSeen[c] = i
      ans += dp

    return ans


# Link: https://leetcode.com/problems/maximize-the-beauty-of-the-garden/description/
class Solution:
  def maximumBeauty(self, flowers: List[int]) -> int:
    ans = -math.inf
    prefix = 0
    flowerToPrefix = collections.defaultdict(int)

    for flower in flowers:
      if flower in flowerToPrefix:
        ans = max(ans, prefix - flowerToPrefix[flower] + flower * 2)
      prefix += max(0, flower)
      flowerToPrefix.setdefault(flower, prefix)

    return ans


# Link: https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/description/
class Solution:
  def minimumDistance(self, word: str) -> int:
    def dist(a: int, b: int) -> int:
      if a == 26:  # the first hovering state
        return 0
      x1, y1 = a // 6, a % 6
      x2, y2 = b // 6, b % 6
      return abs(x1 - x2) + abs(y1 - y2)

    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> int:
      """
      Returns the minimum distance to type the `word`, where the left finger is
      on the i-th letter, the right finger is on the j-th letter, and the
      words[0..k) have been written.
      """
      if k == len(word):
        return 0
      nxt = ord(word[k]) - ord('A')
      moveLeft = dist(i, nxt) + dp(nxt, j, k + 1)
      moveRight = dist(j, nxt) + dp(i, nxt, k + 1)
      return min(moveLeft, moveRight)

    return dp(26, 26, 0)


# Link: https://leetcode.com/problems/check-if-an-original-string-exists-given-two-encoded-strings/description/
class Solution:
  def possiblyEquals(self, s1: str, s2: str) -> bool:
    def getNums(s: str) -> Set[int]:
      nums = {int(s)}
      for i in range(1, len(s)):
        nums |= {x + y for x in getNums(s[:i]) for y in getNums(s[i:])}
      return nums

    def getNextLetterIndex(s: str, i: int) -> int:
      j = i
      while j < len(s) and s[j].isdigit():
        j += 1
      return j

    @functools.lru_cache(None)
    def dp(i: int, j: int, paddingDiff: int) -> bool:
      """
      Returns True if s1[i..n) matches s2[j..n), accounting for the padding
      difference. Here, `paddingDiff` represents the signed padding. A positive
      `paddingDiff` indicates that s1 has an additional number of offset bytes
      compared to s2.
      """
      if i == len(s1) and j == len(s2):
        return paddingDiff == 0
      # Add padding on s1.
      if i < len(s1) and s1[i].isdigit():
        nextLetterIndex = getNextLetterIndex(s1, i)
        for num in getNums(s1[i:nextLetterIndex]):
          if dp(nextLetterIndex, j, paddingDiff + num):
            return True
      # Add padding on s2.
      elif j < len(s2) and s2[j].isdigit():
        nextLetterIndex = getNextLetterIndex(s2, j)
        for num in getNums(s2[j:nextLetterIndex]):
          if dp(i, nextLetterIndex, paddingDiff - num):
            return True
      # `s1` has more padding, so j needs to catch up.
      elif paddingDiff > 0:
        if j < len(s2):
          return dp(i, j + 1, paddingDiff - 1)
      # `s2` has more padding, so i needs to catch up.
      elif paddingDiff < 0:
        if i < len(s1):
          return dp(i + 1, j, paddingDiff + 1)
      # There's no padding difference, so consume the next letter.
      else:  # paddingDiff == 0
        if i < len(s1) and j < len(s2) and s1[i] == s2[j]:
          return dp(i + 1, j + 1, 0)
      return False

    return dp(0, 0, 0)


# Link: https://leetcode.com/problems/burst-balloons/description/
class Solution:
  def maxCoins(self, nums: List[int]) -> int:
    n = len(nums)
    # dp[i][j] := maxCoins(nums[i..j])
    dp = [[0] * (n + 2) for _ in range(n + 2)]

    nums = [1] + nums + [1]

    for d in range(n):
      for i in range(1, n - d + 1):
        j = i + d
        for k in range(i, j + 1):
          dp[i][j] = max(
              dp[i][j],
              dp[i][k - 1] +
              dp[k + 1][j] +
              nums[i - 1] * nums[k] * nums[j + 1])

    return dp[1][n]


# Link: https://leetcode.com/problems/burst-balloons/description/
class Solution:
  def maxCoins(self, nums: List[int]) -> int:
    n = len(nums)
    nums = [1] + nums + [1]

    @functools.lru_cache(None)
    def dp(i: int, j: int) -> int:
      """Returns maxCoins(nums[i..j])."""
      if i > j:
        return 0
      return max(dp(i, k - 1) +
                 dp(k + 1, j) +
                 nums[i - 1] * nums[k] * nums[j + 1]
                 for k in range(i, j + 1))

    return dp(1, n)


# Link: https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def maximumInvitations(self, favorite: List[int]) -> int:
    n = len(favorite)
    sumComponentsLength = 0  # the component: a -> b -> c <-> x <- y
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n
    maxChainLength = [1] * n

    # Build the graph.
    for i, f in enumerate(favorite):
      graph[i].append(f)
      inDegrees[f] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    while q:
      u = q.popleft()
      for v in graph[u]:
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)
        maxChainLength[v] = max(maxChainLength[v], 1 + maxChainLength[u])

    for i in range(n):
      if favorite[favorite[i]] == i:
        # i <-> favorite[i] (the cycle's length = 2)
        sumComponentsLength += maxChainLength[i] + maxChainLength[favorite[i]]

    maxCycleLength = 0  # Cycle: a -> b -> c -> a
    parent = [-1] * n
    seen = set()
    states = [State.kInit] * n

    def findCycle(u: int) -> None:
      nonlocal maxCycleLength
      seen.add(u)
      states[u] = State.kVisiting
      for v in graph[u]:
        if v not in seen:
          parent[v] = u
          findCycle(v)
        elif states[v] == State.kVisiting:
          # Find the cycle's length.
          curr = u
          cycleLength = 1
          while curr != v:
            curr = parent[curr]
            cycleLength += 1
          maxCycleLength = max(maxCycleLength, cycleLength)
      states[u] = State.kVisited

    for i in range(n):
      if i not in seen:
        findCycle(i)

    return max(sumComponentsLength // 2, maxCycleLength)


# Link: https://leetcode.com/problems/dungeon-game/description/
class Solution:
  def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
    m = len(dungeon)
    n = len(dungeon[0])
    dp = [math.inf] * (n + 1)
    dp[n - 1] = 1

    for i in reversed(range(m)):
      for j in reversed(range(n)):
        dp[j] = min(dp[j], dp[j + 1]) - dungeon[i][j]
        dp[j] = max(dp[j], 1)

    return dp[0]


# Link: https://leetcode.com/problems/wildcard-matching/description/
class Solution:
  def isMatch(self, s: str, p: str) -> bool:
    m = len(s)
    n = len(p)
    # dp[i][j] := True if s[0..i) matches p[0..j)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    def isMatch(i: int, j: int) -> bool:
      return i >= 0 and p[j] == '?' or s[i] == p[j]

    for j, c in enumerate(p):
      if c == '*':
        dp[0][j + 1] = dp[0][j]

    for i in range(m):
      for j in range(n):
        if p[j] == '*':
          matchEmpty = dp[i + 1][j]
          matchSome = dp[i][j + 1]
          dp[i + 1][j + 1] = matchEmpty or matchSome
        elif isMatch(i, j):
          dp[i + 1][j + 1] = dp[i][j]

    return dp[m][n]


# Link: https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/description/
class Solution:
  def minArea(self, image: List[List[str]], x: int, y: int) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(image)
    n = len(image[0])
    topLeft = [x, y]
    bottomRight = [x, y]
    q = collections.deque([(x, y)])
    image[x][y] = '2'  # Mark as visited.

    while q:
      i, j = q.popleft()
      for dx, dy in dirs:
        r = i + dx
        c = j + dy
        if r < 0 or r == m or c < 0 or c == n:
          continue
        if image[r][c] != '1':
          continue
        topLeft[0] = min(topLeft[0], r)
        topLeft[1] = min(topLeft[1], c)
        bottomRight[0] = max(bottomRight[0], r)
        bottomRight[1] = max(bottomRight[1], c)
        q.append((r, c))
        image[r][c] = '2'

    width = bottomRight[1] - topLeft[1] + 1
    height = bottomRight[0] - topLeft[0] + 1
    return width * height


# Link: https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/description/
class Solution:
  def minArea(self, image: List[List[str]], x: int, y: int) -> int:
    def firstAnyOne(l: int, r: int, allZeros: Callable[[int], bool]) -> int:
      while l < r:
        m = (l + r) // 2
        if allZeros(m):
          l = m + 1
        else:
          r = m
      return l

    def firstAllZeros(l: int, r: int, allZeros: Callable[[int], bool]) -> int:
      while l < r:
        m = (l + r) // 2
        if allZeros(m):
          r = m
        else:
          l = m + 1
      return l

    def colAllZeros(colIndex: int) -> bool:
      return all(pixel == '0' for pixel in list(zip(*image))[colIndex])

    def rowAllZeros(rowIndex: int) -> bool:
      return all(pixel == '0' for pixel in image[rowIndex])

    x1 = firstAnyOne(0, x, rowAllZeros)
    x2 = firstAllZeros(x + 1, len(image), rowAllZeros)
    y1 = firstAnyOne(0, y, colAllZeros)
    y2 = firstAllZeros(y + 1, len(image[0]), colAllZeros)
    return (x2 - x1) * (y2 - y1)


# Link: https://leetcode.com/problems/apply-operations-on-array-to-maximize-sum-of-squares/description/
class Solution:
  def maxSum(self, nums: List[int], k: int) -> int:
    kMod = 1_000_000_007
    kMaxBit = 30
    ans = 0
    # minIndices[i] := the minimum index in `optimalNums` that the i-th bit
    # should be moved to
    minIndices = [0] * kMaxBit
    optimalNums = [0] * len(nums)

    for num in nums:
      for i in range(kMaxBit):
        if num >> i & 1:
          optimalNums[minIndices[i]] |= 1 << i
          minIndices[i] += 1

    for i in range(k):
      ans += optimalNums[i]**2
      ans %= kMod

    return ans


# Link: https://leetcode.com/problems/largest-palindrome-product/description/
class Solution:
  def largestPalindrome(self, n: int) -> int:
    if n == 1:
      return 9

    kMod = 1337
    upper = pow(10, n) - 1
    lower = pow(10, n - 1) - 1

    for i in range(upper, lower, -1):
      cand = int(str(i) + str(i)[::-1])
      j = upper
      while j * j >= cand:
        if cand % j == 0:
          return cand % kMod
        j -= 1


# Link: https://leetcode.com/problems/maximum-good-people-based-on-statements/description/
class Solution:
  def maximumGood(self, statements: List[List[int]]) -> int:
    n = len(statements)

    def isValid(mask: int) -> bool:
      for i in range(n):
        # The i-th person is bad, so no need to check.
        if (mask >> i & 1) == 0:
          continue
        for j in range(n):
          if statements[i][j] == 2:
            continue
          if statements[i][j] != (mask >> j & 1):
            return False
      return True

    return max(bin(mask).count('1') for mask in range(1 << n) if isValid(mask))


# Link: https://leetcode.com/problems/maximum-good-people-based-on-statements/description/
class Solution:
  def maximumGood(self, statements: List[List[int]]) -> int:
    n = len(statements)
    ans = 0

    def isValid(good: List[int]) -> bool:
      for i, g in enumerate(good):
        if not g:  # The i-th person is bad, so no need to check.
          continue
        for j in range(n):
          if statements[i][j] == 2:
            continue
          if statements[i][j] != good[j]:
            return False
      return True

    def dfs(good: List[int], i: int, count: int) -> None:
      nonlocal ans
      if i == n:
        if isValid(good):
          ans = max(ans, count)
        return

      good.append(0)  # Assume the i-th person is bad.
      dfs(good, i + 1, count)
      good[-1] = 1  # Assume the i-th person is good.
      dfs(good, i + 1, count + 1)
      good.pop()

    dfs([], 0, 0)
    return ans


# Link: https://leetcode.com/problems/number-of-islands-ii/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = [-1] * n
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
  def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    ans = []
    seen = [[False] * n for _ in range(m)]
    uf = UnionFind(m * n)
    count = 0

    def getId(i: int, j: int, n: int) -> int:
      return i * n + j

    for i, j in positions:
      if seen[i][j]:
        ans.append(count)
        continue
      seen[i][j] = True
      id = getId(i, j, n)
      uf.id[id] = id
      count += 1
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        neighborId = getId(x, y, n)
        if uf.id[neighborId] == -1:  # water
          continue
        currentParent = uf.find(id)
        neighborParent = uf.find(neighborId)
        if currentParent != neighborParent:
          uf.unionByRank(currentParent, neighborParent)
          count -= 1
      ans.append(count)

    return ans


# Link: https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree/description/
class Solution:
  def smallestMissingValueSubtree(self, parents: List[int], nums: List[int]) -> List[int]:
    n = len(parents)
    ans = [1] * n
    tree = [[] for _ in range(n)]
    seen = set()
    minMiss = 1

    for i in range(1, n):
      tree[parents[i]].append(i)

    def getNode(nums: List[int]) -> int:
      for i, num in enumerate(nums):
        if num == 1:
          return i
      return -1

    nodeThatsOne = getNode(nums)
    if nodeThatsOne == -1:
      return ans

    u = nodeThatsOne
    prev = -1  # the u that just handled

    def dfs(u: int) -> None:
      seen.add(nums[u])
      for v in tree[u]:
        dfs(v)

    # Upward from `nodeThatsOne` to the root `u`.
    while u != -1:
      for v in tree[u]:
        if v == prev:
          continue
        dfs(v)
      seen.add(nums[u])
      while minMiss in seen:
        minMiss += 1
      ans[u] = minMiss
      prev = u
      u = parents[u]

    return ans


# Link: https://leetcode.com/problems/palindrome-removal/description/
class Solution:
  def minimumMoves(self, arr: List[int]) -> int:
    n = len(arr)
    # dp[i][j] := the minimum number of moves to remove all numbers from arr[i..j]
    dp = [[n] * n for _ in range(n)]

    for i in range(n):
      dp[i][i] = 1

    for i in range(n - 1):
      dp[i][i + 1] = 1 if arr[i] == arr[i + 1] else 2

    for d in range(2, n):
      for i in range(n - d):
        j = i + d
        # Remove arr[i] and arr[j] within the move of removing
        # arr[i + 1..j - 1]
        if arr[i] == arr[j]:
          dp[i][j] = dp[i + 1][j - 1]
        # Try all the possible partitions.
        for k in range(i, j):
          dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j])

    return dp[0][n - 1]


# Link: https://leetcode.com/problems/maximum-xor-with-an-element-from-array/description/
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


class IndexedQuery:
  def __init__(self, queryIndex: int, x: int, m: int):
    self.queryIndex = queryIndex
    self.x = x
    self.m = m

  def __iter__(self):
    yield self.queryIndex
    yield self.x
    yield self.m


class Solution:
  def maximizeXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    ans = [-1] * len(queries)
    maxBit = int(math.log2(max(max(nums), max(x for x, _ in queries))))
    bitTrie = BitTrie(maxBit)

    nums.sort()

    i = 0  # nums' index
    for queryIndex, x, m in sorted([IndexedQuery(i, x, m)
                                    for i, (x, m) in enumerate(queries)],
                                   key=lambda iq: iq.m):
      while i < len(nums) and nums[i] <= m:
        bitTrie.insert(nums[i])
        i += 1
      if i > 0 and nums[i - 1] <= m:
        ans[queryIndex] = bitTrie.getMaxXor(x)

    return ans


# Link: https://leetcode.com/problems/count-stepping-numbers-in-range/description/
class Solution:
  def countSteppingNumbers(self, low: str, high: str) -> int:
    kMod = 1_000_000_007
    low = '0' * (len(high) - len(low)) + low

    @functools.lru_cache(None)
    def dp(i: int, prevDigit: int, isLeadingZero: bool, isTight1: bool, isTight2: bool) -> int:
      """
      Returns the number of valid integers, considering the i-th digit, where
      `prevDigit` is the previous digit, `isTight1` indicates if the current
      digit is tightly bound for `low`, and `isTight2` indicates if the current
      digit is tightly bound for `high`.
      """
      if i == len(high):
        return 1

      res = 0
      minDigit = int(low[i]) if isTight1 else 0
      maxDigit = int(high[i]) if isTight2 else 9

      for d in range(minDigit, maxDigit + 1):
        nextIsTight1 = isTight1 and (d == minDigit)
        nextIsTight2 = isTight2 and (d == maxDigit)
        if isLeadingZero:
          # Can place any digit in [minDigit, maxDigit].
          res += dp(i + 1, d, isLeadingZero and d ==
                    0, nextIsTight1, nextIsTight2)
        elif abs(d - prevDigit) == 1:
          res += dp(i + 1, d, False, nextIsTight1, nextIsTight2)
        res %= kMod

      return res

    return dp(0, -1, True, True, True)


# Link: https://leetcode.com/problems/delivering-boxes-from-storage-to-ports/description/
class Solution:
  def boxDelivering(self, boxes: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int) -> int:
    n = len(boxes)
    # dp[i] := the minimum trips to deliver boxes[0..i) and return to the
    # storage
    dp = [0] * (n + 1)
    trips = 2
    weight = 0

    l = 0
    for r in range(n):
      weight += boxes[r][1]

      # The current box is different from the previous one, need to make one
      # more trip.
      if r > 0 and boxes[r][0] != boxes[r - 1][0]:
        trips += 1

      # Loading boxes[l] in the previous turn is always no bad than loading it
      # in this turn
      while r - l + 1 > maxBoxes or weight > maxWeight or (l < r and dp[l + 1] == dp[l]):
        weight -= boxes[l][1]
        if boxes[l][0] != boxes[l + 1][0]:
          trips -= 1
        l += 1

      #   min trips to deliver boxes[0..r]
      # = min trips to deliver boxes[0..l) + trips to deliver boxes[l..r]
      dp[r + 1] = dp[l] + trips

    return dp[n]


# Link: https://leetcode.com/problems/maximum-score-of-a-node-sequence/description/
class Solution:
  def maximumScore(self, scores: List[int], edges: List[List[int]]) -> int:
    n = len(scores)
    ans = -1
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append((scores[v], v))
      graph[v].append((scores[u], u))

    for i in range(n):
      graph[i] = heapq.nlargest(3, graph[i])

    # To find the target sequence: a - u - v - b, enumerate each edge (u, v),
    # and find a (u's child) and b (v's child). That's why we find the 3
    # children that have the highest scores because one of the 3 children is
    # guaranteed to be valid.
    for u, v in edges:
      for scoreA, a in graph[u]:
        for scoreB, b in graph[v]:
          if a != b and a != v and b != u:
            ans = max(ans, scoreA + scores[u] + scores[v] + scoreB)

    return ans


# Link: https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/description/
class Solution:
  def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
    @functools.lru_cache(None)
    def dp(i: int, k: int) -> int:
      """Returns the maximum value of picking k coins from piles[i..n)."""
      if i == len(piles) or k == 0:
        return 0

      # Pick no coins from the current pile.
      res = dp(i + 1, k)
      val = 0  # the coins picked from the current pile

      # Try to pick 1, 2, ..., k coins from the current pile.
      for j in range(min(len(piles[i]), k)):
        val += piles[i][j]
        res = max(res, val + dp(i + 1, k - j - 1))

      return res

    return dp(0, k)


# Link: https://leetcode.com/problems/maximum-total-beauty-of-the-gardens/description/
class Solution:
  def maximumBeauty(self, flowers: List[int], newFlowers: int, target: int, full: int, partial: int) -> int:
    n = len(flowers)

    # If a garden is already complete, clamp it to the target.
    flowers = [min(flower, target) for flower in flowers]
    flowers.sort()

    # All gardens are complete, so nothing we can do.
    if flowers[0] == target:
      return n * full

    # Having many new flowers maximizes the beauty value.
    if newFlowers >= n * target - sum(flowers):
      return max(n * full, (n - 1) * full + (target - 1) * partial)

    ans = 0
    leftFlowers = newFlowers
    # cost[i] := the cost to make flowers[0..i] the same
    cost = [0] * n

    for i in range(1, n):
      # Plant (flowers[i] - flowers[i - 1]) flowers for flowers[0..i - 1].
      cost[i] = cost[i - 1] + i * (flowers[i] - flowers[i - 1])

    i = n - 1  # flowers' index (flowers[i + 1..n) are complete)
    while flowers[i] == target:
      i -= 1

    while leftFlowers >= 0:
      # To maximize the minimum number of incomplete flowers, we find the first
      # index j that we can't make flowers[0..j] equal to flowers[j], then we
      # know we can make flowers[0..j - 1] equal to flowers[j - 1]. In the
      # meantime, evenly increase each of them to seek a bigger minimum value.
      j = min(i + 1, bisect_right(cost, leftFlowers))
      minIncomplete = flowers[j - 1] + (leftFlowers - cost[j - 1]) // j
      ans = max(ans, (n - 1 - i) * full + minIncomplete * partial)
      leftFlowers -= max(0, target - flowers[i])
      i -= 1

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-operations-to-make-string-sorted/description/
class Solution:
  def makeStringSorted(self, s: str) -> int:
    kMod = 1_000_000_007
    ans = 0
    count = [0] * 26

    @functools.lru_cache(None)
    def fact(i: int) -> int:
      return 1 if i <= 1 else i * fact(i - 1) % kMod

    @functools.lru_cache(None)
    def inv(i: int) -> int:
      return pow(i, kMod - 2, kMod)

    for i, c in enumerate(reversed(s)):
      order = ord(c) - ord('a')
      count[order] += 1
      # count[:order] := s[i] can be any character smaller than c
      # fact(i) := s[i + 1..n - 1] can be any sequence of characters
      perm = sum(count[:order]) * fact(i)
      for j in range(26):
        perm = perm * inv(fact(count[j])) % kMod
      ans = (ans + perm) % kMod

    return ans


# Link: https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/description/
class Solution:
  def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
    ans = sum(jobs)
    times = [0] * k  # times[i] := accumulate time of workers[i]

    # Assign the most time-consuming job first.
    jobs.sort(reverse=True)

    def dfs(s: int) -> None:
      nonlocal ans
      if s == len(jobs):
        ans = min(ans, max(times))
        return
      for i in range(k):
        # There is no need to explore assigning jobs[s] to workers[i] further as
        # it would not yield better results.
        if times[i] + jobs[s] >= ans:
          continue
        times[i] += jobs[s]
        dfs(s + 1)
        times[i] -= jobs[s]
        # It's always non-optimal to have a worker with no jobs.
        if times[i] == 0:
          return

    dfs(0)
    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/description/
class Solution:
  def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
    n = len(passingFees)
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
      graph[u].append((v, w))
      graph[v].append((u, w))

    return self._dijkstra(graph, 0, n - 1, maxTime, passingFees)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, dst: int, maxTime: int, passingFees: List[int]) -> int:
    # cost[i] := the minimum cost to reach the i-th city
    cost = [math.inf for _ in range(len(graph))]
    # dist[i] := the minimum time to reach the i-th city
    dist = [maxTime + 1 for _ in range(len(graph))]

    cost[src] = passingFees[src]
    dist[src] = 0
    minHeap = [(cost[src], dist[src], src)]  # (cost[u], dist[u], u)

    while minHeap:
      currCost, d, u = heapq.heappop(minHeap)
      if u == dst:
        return cost[dst]
      for v, w in graph[u]:
        if d + w > maxTime:
          continue
        # Go from u -> v.
        if currCost + passingFees[v] < cost[v]:
          cost[v] = currCost + passingFees[v]
          dist[v] = d + w
          heapq.heappush(minHeap, (cost[v], dist[v], v))
        elif d + w < dist[v]:
          dist[v] = d + w
          heapq.heappush(minHeap, (currCost + passingFees[v], dist[v], v))

    return -1


# Link: https://leetcode.com/problems/data-stream-as-disjoint-intervals/description/
from sortedcontainers import SortedDict


class SummaryRanges:
  def __init__(self):
    self.intervals = SortedDict()  # {start: (start, end)}

  def addNum(self, val: int) -> None:
    if val in self.intervals:
      return

    lo = self._lowerKey(val)
    hi = self._higherKey(val)

    # {lo, map[lo][1]} + val + {hi, map[hi][1]} = {lo, map[hi][1]}
    if lo >= 0 and hi >= 0 and self.intervals[lo][1] + 1 == val and val + 1 == hi:
      self.intervals[lo][1] = self.intervals[hi][1]
      del self.intervals[hi]
      # {lo, map[lo][1]} + val = {lo, val}
      # Prevent adding duplicate entry by using '>=' instead of '=='.
    elif lo >= 0 and self.intervals[lo][1] + 1 >= val:
      self.intervals[lo][1] = max(self.intervals[lo][1], val)
    elif hi >= 0 and val + 1 == hi:
      # val + {hi, map[hi][1]} = {val, map[hi][1]}
      self.intervals[val] = [val, self.intervals[hi][1]]
      del self.intervals[hi]
    else:
      self.intervals[val] = [val, val]

  def getIntervals(self) -> List[List[int]]:
    return list(self.intervals.values())

  def _lowerKey(self, key: int):
    """Returns the maximum key in `self.intervals` < `key`."""
    i = self.intervals.bisect_left(key)
    if i == 0:
      return -1
    return self.intervals.peekitem(i - 1)[0]

  def _higherKey(self, key: int):
    """Returns the minimum key in `self.intervals` < `key`."""
    i = self.intervals.bisect_right(key)
    if i == len(self.intervals):
      return -1
    return self.intervals.peekitem(i)[0]


# Link: https://leetcode.com/problems/text-justification/description/
class Solution:
  def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    ans = []
    row = []
    rowLetters = 0

    for word in words:
      # If we place the word in this row, it will exceed the maximum width.
      # Therefore, we cannot put the word in this row and have to pad spaces
      # for each word in this row.
      if rowLetters + len(word) + len(row) > maxWidth:
        for i in range(maxWidth - rowLetters):
          row[i % (len(row) - 1 or 1)] += ' '
        ans.append(''.join(row))
        row = []
        rowLetters = 0
      row.append(word)
      rowLetters += len(word)

    return ans + [' '.join(row).ljust(maxWidth)]


# Link: https://leetcode.com/problems/minimum-swaps-to-make-sequences-increasing/description/
class Solution:
  def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
    keepAt = [math.inf] * len(nums1)
    swapAt = [math.inf] * len(nums1)
    keepAt[0] = 0
    swapAt[0] = 1

    for i in range(1, len(nums1)):
      if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
        keepAt[i] = keepAt[i - 1]
        swapAt[i] = swapAt[i - 1] + 1
      if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
        keepAt[i] = min(keepAt[i], swapAt[i - 1])
        swapAt[i] = min(swapAt[i], keepAt[i - 1] + 1)

    return min(keepAt[-1], swapAt[-1])


# Link: https://leetcode.com/problems/number-of-visible-people-in-a-queue/description/
class Solution:
  def canSeePersonsCount(self, heights: List[int]) -> List[int]:
    ans = [0] * len(heights)
    stack = []

    for i, height in enumerate(heights):
      while stack and heights[stack[-1]] <= height:
        ans[stack.pop()] += 1
      if stack:
        ans[stack[-1]] += 1
      stack.append(i)

    return ans


# Link: https://leetcode.com/problems/find-the-number-of-ways-to-place-people-ii/description/
class Solution:
  # Same as 3025. Find the Number of Ways to Place People I
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


# Link: https://leetcode.com/problems/sum-of-floored-pairs/description/
class Solution:
  def sumOfFlooredPairs(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    kMax = max(nums)
    ans = 0
    count = [0] * (kMax + 1)

    for num in nums:
      count[num] += 1

    for i in range(1, kMax + 1):
      count[i] += count[i - 1]

    for i in range(1, kMax + 1):
      if count[i] > count[i - 1]:
        summ = 0
        j = 1
        while i * j <= kMax:
          lo = i * j - 1
          hi = i * (j + 1) - 1
          summ += (count[min(hi, kMax)] - count[lo]) * j
          j += 1
        ans += summ * (count[i] - count[i - 1])
        ans %= kMod

    return ans


# Link: https://leetcode.com/problems/sort-array-by-moving-items-to-empty-space/description/
class Solution:
  def sortArray(self, nums: List[int]) -> int:
    n = len(nums)
    numToIndex = [0] * n

    for i, num in enumerate(nums):
      numToIndex[num] = i

    def minOps(numToIndex: List[int], zeroInBeginning: bool) -> int:
      ops = 0
      num = 1
      # If zeroInBeginning, the correct index of each num is num.
      # If not zeroInBeginning, the correct index of each num is num - 1.
      offset = 0 if zeroInBeginning else 1
      while num < n:
        # 0 is in the correct index, so swap 0 with the first `numInWrongIndex`.
        if zeroInBeginning and numToIndex[0] == 0 or \
                not zeroInBeginning and numToIndex[0] == n - 1:
          while numToIndex[num] == num - offset:  # num is in correct position
            num += 1
            if num == n:
              return ops
          numInWrongIndex = num
        # 0 is in the wrong index. e.g. numToIndex[0] == 2, that means 2 is not
        # in nums[2] because nums[2] == 0.
        else:
          numInWrongIndex = numToIndex[0] + offset
        numToIndex[0], numToIndex[numInWrongIndex] = \
            numToIndex[numInWrongIndex], numToIndex[0]
        ops += 1

    return min(minOps(numToIndex.copy(), True),
               minOps(numToIndex.copy(), False))


# Link: https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/description/
class Solution:
  def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
    n = len(nums)
    xors = functools.reduce(lambda x, y: x ^ y, nums)
    subXors = nums[:]
    tree = [[] for _ in range(n)]
    children = [{i} for i in range(n)]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int, parent: int) -> Tuple[int, Set[int]]:
      for v in tree[u]:
        if v == parent:
          continue
        vXor, vChildren = dfs(v, u)
        subXors[u] ^= vXor
        children[u] |= vChildren
      return subXors[u], children[u]

    dfs(0, -1)

    ans = math.inf
    for i in range(len(edges)):
      a, b = edges[i]
      if b in children[a]:
        a, b = b, a
      for j in range(i):
        c, d = edges[j]
        if d in children[c]:
          c, d = d, c

        if c in children[a] and a != c:
          cands = [subXors[c], subXors[a] ^ subXors[c], xors ^ subXors[a]]
        elif a in children[c] and a != c:
          cands = [subXors[a], subXors[c] ^ subXors[a], xors ^ subXors[c]]
        else:
          cands = [subXors[a], subXors[c], xors ^ subXors[a] ^ subXors[c]]
        ans = min(ans, max(cands) - min(cands))

    return ans


# Link: https://leetcode.com/problems/add-edges-to-make-degrees-of-all-nodes-even/description/
class Solution:
  def isPossible(self, n: int, edges: List[List[int]]) -> bool:
    graph = [set() for _ in range(n)]

    for u, v in edges:
      graph[u - 1].add(v - 1)
      graph[v - 1].add(u - 1)

    oddNodes = [i for i, neighbor in enumerate(graph) if len(neighbor) & 1]
    if not oddNodes:
      return True
    if len(oddNodes) == 2:
      a, b = oddNodes
      return any(a not in graph[i] and b not in graph[i] for i in range(n))
    if len(oddNodes) == 4:
      a, b, c, d = oddNodes
      return (b not in graph[a] and d not in graph[c]) or \
          (c not in graph[a] and d not in graph[b]) or \
          (d not in graph[a] and c not in graph[b])
    return False


# Link: https://leetcode.com/problems/maximum-path-quality-of-a-graph/description/
class Solution:
  def maximalPathQuality(self, values: List[int], edges: List[List[int]], maxTime: int) -> int:
    ans = 0
    graph = [[] for _ in range(len(values))]
    # (node, quality, remainingTime, seen)
    q = collections.deque([(0, values[0], maxTime, {0})])

    for u, v, time in edges:
      graph[u].append((v, time))
      graph[v].append((u, time))

    while q:
      u, quality, remainingTime, seen = q.popleft()
      if u == 0:
        ans = max(ans, quality)
      for v, time in graph[u]:
        if time <= remainingTime:
          q.append(
              (v, quality + values[v] * (v not in seen), remainingTime - time, seen | set([v])))

    return ans


# Link: https://leetcode.com/problems/maximum-path-quality-of-a-graph/description/
class Solution:
  def maximalPathQuality(self, values: List[int], edges: List[List[int]], maxTime: int) -> int:
    n = len(values)
    ans = 0
    graph = [[] for _ in range(n)]
    seen = [0] * n
    seen[0] = 1

    for u, v, time in edges:
      graph[u].append((v, time))
      graph[v].append((u, time))

    def dfs(u: int, quality: int, remainingTime: int):
      nonlocal ans
      if u == 0:
        ans = max(ans, quality)
      for v, time in graph[u]:
        if time > remainingTime:
          continue
        newQuality = quality + values[v] * (seen[v] == 0)
        seen[v] += 1
        dfs(v, newQuality, remainingTime - time)
        seen[v] -= 1

    dfs(0, values[0], maxTime)
    return ans


# Link: https://leetcode.com/problems/minimum-time-to-eat-all-grains/description/
class Solution:
  def minimumTime(self, hens: List[int], grains: List[int]) -> int:
    hens.sort()
    grains.sort()

    def canEat(time: int) -> bool:
      """Returns True if `hens` can eat all `grains` within `time`."""
      i = 0  # grains[i] := next grain to be ate
      for hen in hens:
        rightMoves = time
        if grains[i] < hen:
          # `hen` needs go back to eat `grains[i]`.
          leftMoves = hen - grains[i]
          if leftMoves > time:
            return False
          leftThenRight = time - 2 * leftMoves
          rightThenLeft = (time - leftMoves) // 2
          rightMoves = max(0, leftThenRight, rightThenLeft)
        i = bisect.bisect_right(grains, hen + rightMoves)
        if i == len(grains):
          return True
      return False

    maxMoves = int(1.5 * (max(hens + grains) - min(hens + grains)))
    return bisect.bisect_left(range(maxMoves), True, key=lambda m: canEat(m))


# Link: https://leetcode.com/problems/distribute-candies-among-children-iii/description/
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


# Link: https://leetcode.com/problems/range-sum-query-2d-mutable/description/
class FenwickTree:
  def __init__(self, m: int, n: int):
    self.sums = [[0] * (n + 1) for _ in range(m + 1)]

  def update(self, row: int, col: int, delta: int) -> None:
    i = row
    while i < len(self.sums):
      j = col
      while j < len(self.sums[0]):
        self.sums[i][j] += delta
        j += FenwickTree.lowbit(j)
      i += FenwickTree.lowbit(i)

  def get(self, row: int, col: int) -> int:
    summ = 0
    i = row
    while i > 0:
      j = col
      while j > 0:
        summ += self.sums[i][j]
        j -= FenwickTree.lowbit(j)
      i -= FenwickTree.lowbit(i)
    return summ

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i


class NumMatrix:
  def __init__(self, matrix: List[List[int]]):
    self.matrix = matrix
    self.tree = FenwickTree(len(matrix), len(matrix[0]))

    for i in range(len(matrix)):
      for j, val in enumerate(matrix[i]):
        self.tree.update(i + 1, j + 1, val)

  def update(self, row: int, col: int, val: int) -> None:
    self.tree.update(row + 1, col + 1, val - self.matrix[row][col])
    self.matrix[row][col] = val

  def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
    return self.tree.get(row2 + 1, col2 + 1) - self.tree.get(row1, col2 + 1) - \
        self.tree.get(row2 + 1, col1) + self.tree.get(row1, col1)


# Link: https://leetcode.com/problems/largest-rectangle-in-histogram/description/
class Solution:
  def largestRectangleArea(self, heights: List[int]) -> int:
    ans = 0
    stack = []

    for i in range(len(heights) + 1):
      while stack and (i == len(heights) or heights[stack[-1]] > heights[i]):
        h = heights[stack.pop()]
        w = i - stack[-1] - 1 if stack else i
        ans = max(ans, h * w)
      stack.append(i)

    return ans


# Link: https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/description/
class Solution:
  def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
    n = len(nums) - k + 1
    # sums[i] := sum(nums[i..i + k))
    sums = [0] * n
    # l[i] := the index in [0..i] that has the maximum sums[i]
    l = [0] * n
    # r[i] := the index in [i..n) that has the maximum sums[i]
    r = [0] * n

    summ = 0
    for i, num in enumerate(nums):
      summ += num
      if i >= k:
        summ -= nums[i - k]
      if i >= k - 1:
        sums[i - k + 1] = summ

    maxIndex = 0
    for i in range(n):
      if sums[i] > sums[maxIndex]:
        maxIndex = i
      l[i] = maxIndex

    maxIndex = n - 1
    for i in range(n - 1, -1, -1):
      if sums[i] >= sums[maxIndex]:
        maxIndex = i
      r[i] = maxIndex

    ans = [-1, -1, -1]

    for i in range(k, n - k):
      if ans[0] == -1 or sums[ans[0]] + sums[ans[1]] + sums[ans[2]] <\
              sums[l[i - k]] + sums[i] + sums[r[i + k]]:
        ans[0] = l[i - k]
        ans[1] = i
        ans[2] = r[i + k]

    return ans


# Link: https://leetcode.com/problems/valid-arrangement-of-pairs/description/
class Solution:
  def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
    ans = []
    graph = collections.defaultdict(list)
    outDegree = collections.Counter()
    inDegrees = collections.Counter()

    for start, end in pairs:
      graph[start].append(end)
      outDegree[start] += 1
      inDegrees[end] += 1

    def getStartNode() -> int:
      for u in graph.keys():
        if outDegree[u] - inDegrees[u] == 1:
          return u
      return pairs[0][0]  # Arbitrarily choose a node.

    def euler(u: int) -> None:
      stack = graph[u]
      while stack:
        v = stack.pop()
        euler(v)
        ans.append([u, v])

    euler(getStartNode())
    return ans[::-1]


# Link: https://leetcode.com/problems/divide-array-into-increasing-sequences/description/
class Solution:
  def canDivideIntoSubsequences(self, nums: List[int], k: int) -> bool:
    # Find the number with the maxFreq, we need at least maxFreq * k elements
    # e.g. nums = [1, 2, 2, 3, 4], we have maxFreq = 2 (two 2s), so we have to
    # Split nums into two subsequences say k = 3, the minimum length of nums is 2 x
    # 3 = 6, which is impossible if len(nums) = 5
    return len(nums) >= k * max(Counter(nums).values())


# Link: https://leetcode.com/problems/maximum-students-taking-exam/description/
class Solution:
  def maxStudents(self, seats: List[List[str]]) -> int:
    m = len(seats)
    n = len(seats[0])
    dirs = ((-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1))
    seen = [[0] * n for _ in range(m)]
    match = [[-1] * n for _ in range(m)]

    def dfs(i: int, j: int, sessionId: int) -> int:
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if seats[x][y] != '.' or seen[x][y] == sessionId:
          continue
        seen[x][y] = sessionId
        if match[x][y] == -1 or dfs(*divmod(match[x][y], n), sessionId):
          match[x][y] = i * n + j
          match[i][j] = x * n + y
          return 1
      return 0

    def hungarian() -> int:
      count = 0
      for i in range(m):
        for j in range(n):
          if seats[i][j] == '.' and match[i][j] == -1:
            sessionId = i * n + j
            seen[i][j] = sessionId
            count += dfs(i, j, sessionId)
      return count

    return sum(seats[i][j] == '.' for i in range(m)
               for j in range(n)) - hungarian()


# Link: https://leetcode.com/problems/word-break-ii/description/
class Solution:
  def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
    wordSet = set(wordDict)

    @functools.lru_cache(None)
    def wordBreak(s: str) -> List[str]:
      ans = []

      # 1 <= len(prefix) < len(s)
      for i in range(1, len(s)):
        prefix = s[0:i]
        suffix = s[i:]
        if prefix in wordSet:
          for word in wordBreak(suffix):
            ans.append(prefix + ' ' + word)

      # `wordSet` contains the whole string s, so don't add any space.
      if s in wordSet:
        ans.append(s)

      return ans

    return wordBreak(s)


# Link: https://leetcode.com/problems/minimum-time-to-build-blocks/description/
class Solution:
  def minBuildTime(self, blocks: List[int], split: int) -> int:
    minHeap = blocks.copy()
    heapify(minHeap)

    while len(minHeap) > 1:
      heapq.heappop(minHeap)  # the minimum
      x = heapq.heappop(minHeap)  # the second minimum
      heapq.heappush(minHeap, x + split)

    return minHeap[0]


# Link: https://leetcode.com/problems/graph-connectivity-with-threshold/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.rank = [0] * n

  def unionByRank(self, u: int, v: int) -> bool:
    i = self.find(u)
    j = self.find(v)
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

  def find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self.find(self.id[u])
    return self.id[u]


class Solution:
  def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
    uf = UnionFind(n + 1)

    for z in range(threshold + 1, n + 1):
      for x in range(z * 2, n + 1, z):
        uf.unionByRank(z, x)

    return [uf.find(a) == uf.find(b) for a, b in queries]


# Link: https://leetcode.com/problems/merge-bsts-to-create-single-bst/description/
class Solution:
  def canMerge(self, trees: List[TreeNode]) -> Optional[TreeNode]:
    valToNode = {}  # {val: node}
    count = collections.Counter()  # {val: freq}

    for tree in trees:
      valToNode[tree.val] = tree
      count[tree.val] += 1
      if tree.left:
        count[tree.left.val] += 1
      if tree.right:
        count[tree.right.val] += 1

    def isValidBST(tree: Optional[TreeNode], minNode: Optional[TreeNode], maxNode: Optional[TreeNode]) -> bool:
      if not tree:
        return True
      if minNode and tree.val <= minNode.val:
        return False
      if maxNode and tree.val >= maxNode.val:
        return False
      if not tree.left and not tree.right and tree.val in valToNode:
        val = tree.val
        tree.left = valToNode[val].left
        tree.right = valToNode[val].right
        del valToNode[val]

      return isValidBST(tree.left, minNode, tree) and isValidBST(tree.right, tree, maxNode)

    for tree in trees:
      if count[tree.val] == 1:
        if isValidBST(tree, None, None) and len(valToNode) <= 1:
          return tree
        return None

    return None


# Link: https://leetcode.com/problems/string-transforms-into-another-string/description/
class Solution:
  def canConvert(self, str1: str, str2: str) -> bool:
    if str1 == str2:
      return True

    mappings = {}

    for a, b in zip(str1, str2):
      if mappings.get(a, b) != b:
        return False
      mappings[a] = b

    # No letter in the str1 maps to > 1 letter in the str2 and there is at
    # lest one temporary letter can break any loops.
    return len(set(str2)) < 26


# Link: https://leetcode.com/problems/minimum-degree-of-a-connected-trio-in-a-graph/description/
class Solution:
  def minTrioDegree(self, n: int, edges: List[List[int]]) -> int:
    ans = math.inf
    graph = [set() for _ in range(n)]
    degrees = [0] * n

    for u, v in edges:
      u -= 1
      v -= 1
      # Store the mapping from `min(u, v)` to `max(u, v)` to speed up.
      graph[min(u, v)].add(max(u, v))
      degrees[u] += 1
      degrees[v] += 1

    for u in range(n):
      for v in graph[u]:
        for w in graph[u]:
          if w in graph[v]:
            ans = min(ans, degrees[u] + degrees[v] + degrees[w] - 6)

    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/course-schedule-iii/description/
class Solution:
  def scheduleCourse(self, courses: List[List[int]]) -> int:
    time = 0
    maxHeap = []

    for duration, lastDay in sorted(courses, key=lambda x: x[1]):
      heapq.heappush(maxHeap, -duration)
      time += duration
      # If the current course cannot be taken, check if it can be swapped with
      # a previously taken course that has a larger duration to increase the
      # time available to take upcoming courses.
      if time > lastDay:
        time += heapq.heappop(maxHeap)

    return len(maxHeap)


# Link: https://leetcode.com/problems/pizza-with-3n-slices/description/
class Solution:
  def maxSizeSlices(self, slices: List[int]) -> int:
    @functools.lru_cache(None)
    def dp(i: int, j: int, k: int) -> int:
      """
      Returns the maximum the sum of slices if you can pick k slices from
      slices[i..j).
      """
      if k == 1:
        return max(slices[i:j])
      # Note that j - i is not the number of all the left slices. Since you
      # Might have chosen not to take a slice in a previous step, there would be
      # Leftovers outside [i:j]. If you take slices[i], one of the slices your
      # Friends take will be outside of [i:j], so the length of [i:j] is reduced
      # By 2 instead of 3. Therefore, the minimum # Is 2 * k - 1 (the last step only
      # Requires one slice).
      if j - i < 2 * k - 1:
        return -math.inf
      return max(slices[i] + dp(i + 2, j, k - 1),
                 dp(i + 1, j, k))

    k = len(slices) // 3
    return max(dp(0, len(slices) - 1, k),
               dp(1, len(slices), k))


# Link: https://leetcode.com/problems/minimum-time-to-kill-all-monsters/description/
class Solution:
  def minimumTime(self, power: List[int]) -> int:
    n = len(power)
    maxMask = 1 << n
    # dp[i] := the minimum number of days needed to defeat the monsters, where
    # i is the bitmask of the monsters
    dp = [math.inf] * maxMask
    dp[0] = 0

    for mask in range(1, maxMask):
      currentGain = mask.bit_count()
      for i in range(n):
        if mask >> i & 1:
          dp[mask] = min(dp[mask], dp[mask & ~(1 << i)] +
                         int(math.ceil(power[i] / currentGain)))

    return dp[-1]


# Link: https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/description/
class Solution:
  def minSkips(self, dist: List[int], speed: int, hoursBefore: int) -> int:
    kInf = 10**7
    kEps = 1e-9
    n = len(dist)
    # dp[i][j] := the minimum time, where i is the number of roads we traversed
    # so far and j is the number of skips we did
    dp = [[kInf] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i, d in enumerate(dist, 1):
      dp[i][0] = math.ceil(dp[i - 1][0] + d / speed - kEps)
      for j in range(1, i + 1):
        dp[i][j] = min(dp[i - 1][j - 1] + d / speed,
                       math.ceil(dp[i - 1][j] + d / speed - kEps))

    for j, time in enumerate(dp[-1]):
      if time <= hoursBefore:
        return j

    return -1


# Link: https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/
class Solution:
  def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
    maxSum = sum(max(num, num ^ k) for num in nums)
    changedCount = sum((num ^ k) > num for num in nums)
    if changedCount % 2 == 0:
      return maxSum
    minChangeDiff = min(abs(num - (num ^ k)) for num in nums)
    return maxSum - minChangeDiff


# Link: https://leetcode.com/problems/maximize-the-minimum-powered-city/description/
class Solution:
  def maxPower(self, stations: List[int], r: int, k: int) -> int:
    n = len(stations)
    left = min(stations)
    right = sum(stations) + k + 1

    # Returns true if each city can have at least `minPower`.
    def check(stations: List[int], additionalStations: int, minPower: int) -> bool:
      # Initilaize `power` as the 0-th city's power - stations[r].
      power = sum(stations[:r])

      for i in range(n):
        if i + r < n:
          power += stations[i + r]  # `power` = sum(stations[i - r..i + r]).
        if power < minPower:
          requiredPower = minPower - power
          # There're not enough stations to plant.
          if requiredPower > additionalStations:
            return False
          # Greedily plant `requiredPower` power stations in the farthest place
          # to cover as many cities as possible.
          stations[min(n - 1, i + r)] += requiredPower
          additionalStations -= requiredPower
          power += requiredPower
        if i - r >= 0:
          power -= stations[i - r]

      return True

    while left < right:
      mid = (left + right) // 2
      if check(stations.copy(), k, mid):
        left = mid + 1
      else:
        right = mid

    return left - 1


# Link: https://leetcode.com/problems/make-array-non-decreasing-or-non-increasing/description/
class Solution:
  def convertArray(self, nums: List[int]) -> int:
    def cost(nums: List[int]) -> int:
      ans = 0
      minHeap = []

      # Greedily make `nums` non-increasing.
      for num in nums:
        if minHeap and minHeap[0] < num:
          ans += num - heapq.heappushpop(minHeap, num)
        heapq.heappush(minHeap, num)

      return ans

    return min(cost(nums), cost([-num for num in nums]))


# Link: https://leetcode.com/problems/count-all-possible-routes/description/
class Solution:
  def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(i: int, fuel: int) -> int:
      """
      Returns the number of ways to reach the `finish` city from the i-th city
      with `fuel` fuel.
      """
      if fuel < 0:
        return 0

      res = 1 if i == finish else 0
      for j in range(len(locations)):
        if j == i:
          continue
        res += dp(j, fuel - abs(locations[i] - locations[j]))
        res %= kMod

      return res

    return dp(start, fuel)


# Link: https://leetcode.com/problems/count-all-possible-routes/description/
class Solution:
  def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
    kMod = 1_000_000_007
    n = len(locations)
    # dp[i][j] := the number of ways to reach the `finish` city from the i-th
    # city with `j` fuel
    dp = [[0] * (fuel + 1) for _ in range(n)]

    for f in range(fuel + 1):
      dp[finish][f] = 1

    for f in range(fuel + 1):
      for i in range(n):
        for j in range(n):
          if i == j:
            continue
          requiredFuel = abs(locations[i] - locations[j])
          if requiredFuel <= f:
            dp[i][f] += dp[j][f - requiredFuel]
            dp[i][f] %= kMod

    return dp[start][fuel]


# Link: https://leetcode.com/problems/minimum-cost-to-cut-a-stick/description/
class Solution:
  def minCost(self, n: int, cuts: List[int]) -> int:
    A = sorted([0] + cuts + [n])

    dp = [[0] * len(A) for _ in range(len(A))]

    for d in range(2, len(A)):
      for i in range(len(A) - d):
        j = i + d
        dp[i][j] = math.inf
        for k in range(i + 1, j):
          dp[i][j] = min(dp[i][j], A[j] - A[i] + dp[i][k] + dp[k][j])

    return dp[0][len(A) - 1]


# Link: https://leetcode.com/problems/minimum-cost-to-cut-a-stick/description/
class Solution:
  def minCost(self, n: int, cuts: List[int]) -> int:
    A = sorted([0] + cuts + [n])

    @functools.lru_cache(None)
    def dp(i, j):
      if j - i <= 1:
        return 0

      return min(A[j] - A[i] + dp(i, k) + dp(k, j) for k in range(i + 1, j))

    return dp(0, len(A) - 1)


# Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/
class Solution:
  def maxProfit(self, prices: List[int]) -> int:
    sellTwo = 0
    holdTwo = -math.inf
    sellOne = 0
    holdOne = -math.inf

    for price in prices:
      sellTwo = max(sellTwo, holdTwo + price)
      holdTwo = max(holdTwo, sellOne - price)
      sellOne = max(sellOne, holdOne + price)
      holdOne = max(holdOne, -price)

    return sellTwo


# Link: https://leetcode.com/problems/count-beautiful-substrings-ii/description/
class Solution:
  # Same as 2947. Count Beautiful Substrings I
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


# Link: https://leetcode.com/problems/find-number-of-coins-to-place-in-tree-nodes/description/
class ChildCost:
  def __init__(self, cost: int):
    self.numNodes = 1
    self.maxPosCosts = [cost] if cost > 0 else []
    self.minNegCosts = [cost] if cost < 0 else []

  def update(self, childCost: 'ChildCost') -> None:
    self.numNodes += childCost.numNodes
    self.maxPosCosts.extend(childCost.maxPosCosts)
    self.minNegCosts.extend(childCost.minNegCosts)
    self.maxPosCosts.sort(reverse=True)
    self.minNegCosts.sort()
    self.maxPosCosts = self.maxPosCosts[:3]
    self.minNegCosts = self.minNegCosts[:2]

  def maxProduct(self) -> int:
    if self.numNodes < 3:
      return 1
    if not self.maxPosCosts:
      return 0
    res = 0
    if len(self.maxPosCosts) == 3:
      res = self.maxPosCosts[0] * self.maxPosCosts[1] * self.maxPosCosts[2]
    if len(self.minNegCosts) == 2:
      res = max(res,
                self.minNegCosts[0] * self.minNegCosts[1] * self.maxPosCosts[0])
    return res


class Solution:
  def placedCoins(self, edges: List[List[int]], cost: List[int]) -> List[int]:
    n = len(cost)
    ans = [0] * n
    tree = [[] for _ in range(n)]

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    def dfs(u: int, prev: int) -> None:
      res = ChildCost(cost[u])
      for v in tree[u]:
        if v != prev:
          res.update(dfs(v, u))
      ans[u] = res.maxProduct()
      return res

    dfs(0, -1)
    return ans


# Link: https://leetcode.com/problems/string-compression-ii/description/
class Solution:
  def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
    def getLength(maxFreq: int) -> int:
      """Returns the length to compress `maxFreq`."""
      if maxFreq == 1:
        return 1  # c
      if maxFreq < 10:
        return 2  # [1-9]c
      if maxFreq < 100:
        return 3  # [1-9][0-9]c
      return 4    # [1-9][0-9][0-9]c

    @functools.lru_cache(None)
    def dp(i: int, k: int) -> int:
      """Returns the length of optimal dp of s[i..n) with at most k deletion."""
      if k < 0:
        return math.inf
      if i == len(s) or len(s) - i <= k:
        return 0

      ans = math.inf
      maxFreq = 0  # the maximum frequency in s[i..j]
      count = collections.Counter()

      # Make letters in s[i..j] be the same.
      # Keep the letter that has the maximum frequency in this range and remove
      # the other letters.
      for j in range(i, len(s)):
        count[s[j]] += 1
        maxFreq = max(maxFreq, count[s[j]])
        ans = min(ans, getLength(maxFreq) +
                  dp(j + 1, k - (j - i + 1 - maxFreq)))

      return ans

    return dp(0, k)


# Link: https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/description/
class Solution:
  def numberOfWays(self, corridor: str) -> int:
    kMod = 1_000_000_007
    ans = 1
    prevSeat = -1
    numSeats = 0

    for i, c in enumerate(corridor):
      if c == 'S':
        numSeats += 1
        if numSeats > 2 and numSeats & 1:
          ans = ans * (i - prevSeat) % kMod
        prevSeat = i

    return ans if numSeats > 1 and numSeats % 2 == 0 else 0


# Link: https://leetcode.com/problems/tag-validator/description/
class Solution:
  def isValid(self, code: str) -> bool:
    if code[0] != '<' or code[-1] != '>':
      return False

    containsTag = False
    stack = []

    def isValidCdata(s: str) -> bool:
      return s.find('[CDATA[') == 0

    def isValidTagName(tagName: str, isEndTag: bool) -> bool:
      nonlocal containsTag
      if not tagName or len(tagName) > 9:
        return False
      if any(not c.isupper() for c in tagName):
        return False

      if isEndTag:
        return stack and stack.pop() == tagName

      containsTag = True
      stack.append(tagName)
      return True

    i = 0
    while i < len(code):
      if not stack and containsTag:
        return False
      if code[i] == '<':
        # It's inside a tag, so check if it's a cdata.
        if stack and code[i + 1] == '!':
          closeIndex = code.find(']]>', i + 2)
          if closeIndex == -1 or not isValidCdata(code[i + 2:closeIndex]):
            return False
        elif code[i + 1] == '/':  # the end tag
          closeIndex = code.find('>', i + 2)
          if closeIndex == -1 or not isValidTagName(code[i + 2:closeIndex], True):
            return False
        else:  # the start tag
          closeIndex = code.find('>', i + 1)
          if closeIndex == -1 or not isValidTagName(code[i + 1:closeIndex], False):
            return False
        i = closeIndex
      i += 1

    return not stack and containsTag


# Link: https://leetcode.com/problems/count-special-integers/description/
class Solution:
  # Same as 1012. Numbers With Repeated Digits
  def countSpecialNumbers(self, n: int) -> int:
    s = str(n)

    @functools.lru_cache(None)
    def dp(i: int, used: int, isTight: bool) -> int:
      """
      Returns the number of special integers, considering the i-th digit, where
      `used` is the bitmask of the used digits, and `isTight` indicates if the
      current digit is tightly bound.
      """
      if i == len(s):
        return 1

      res = 0

      maxDigit = int(s[i]) if isTight else 9
      for d in range(maxDigit + 1):
        # `d` is used.
        if used >> d & 1:
          continue
        # Use `d` now.
        nextIsTight = isTight and (d == maxDigit)
        if used == 0 and d == 0:  # Don't count leading 0s as used.
          res += dp(i + 1, used, nextIsTight)
        else:
          res += dp(i + 1, used | 1 << d, nextIsTight)

      return res

    return dp(0, 0, True) - 1  # - 0


# Link: https://leetcode.com/problems/make-array-empty/description/
class Solution:
  def countOperationsToEmptyArray(self, nums: List[int]) -> int:
    n = len(nums)
    ans = n
    numToIndex = {}

    for i, num in enumerate(nums):
      numToIndex[num] = i

    nums.sort()

    for i in range(1, n):
      # On the i-th step we've already removed the i - 1 smallest numbers and
      # can ignore them. If an element nums[i] has smaller index in origin
      # array than nums[i - 1], we should rotate the whole left array n - i
      # times to set nums[i] element on the first position.
      if numToIndex[nums[i]] < numToIndex[nums[i - 1]]:
        ans += n - i

    return ans


# Link: https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/description/
class Solution:
  def waysToBuildRooms(self, prevRoom: List[int]) -> int:
    kMod = 1_000_000_007
    graph = collections.defaultdict(list)

    for i, prev in enumerate(prevRoom):
      graph[prev].append(i)

    def dfs(node: int) -> Tuple[int, int]:
      if not graph[node]:
        return 1, 1

      ans = 1
      l = 0

      for child in graph[node]:
        temp, r = dfs(child)
        ans = (ans * temp * math.comb(l + r, r)) % kMod
        l += r

      return ans, l + 1

    return dfs(0)[0]


# Link: https://leetcode.com/problems/maximum-running-time-of-n-computers/description/
class Solution:
  def maxRunTime(self, n: int, batteries: List[int]) -> int:
    summ = sum(batteries)

    batteries.sort()

    # The maximum battery is greater than the average, so it can last forever.
    # Reduce the problem from size n to size n - 1.
    while batteries[-1] > summ // n:
      summ -= batteries.pop()
      n -= 1

    # If the maximum battery <= average running time, it won't be waste, and so
    # do smaller batteries.
    return summ // n


# Link: https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
class Solution:
  def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    m = len(matrix)
    n = len(matrix[0])

    @functools.lru_cache(None)
    def dfs(i: int, j: int, prev: int) -> int:
      if i < 0 or i == m or j < 0 or j == n:
        return 0
      if matrix[i][j] <= prev:
        return 0

      curr = matrix[i][j]
      return 1 + max(dfs(i + 1, j, curr),
                     dfs(i - 1, j, curr),
                     dfs(i, j + 1, curr),
                     dfs(i, j - 1, curr))

    return max(dfs(i, j, -math.inf) for i in range(m) for j in range(n))


# Link: https://leetcode.com/problems/longest-valid-parentheses/description/
class Solution:
  def longestValidParentheses(self, s: str) -> int:
    s2 = ')' + s
    # dp[i] := the length of the longest valid parentheses in the substring
    # s2[1..i]
    dp = [0] * len(s2)

    for i in range(1, len(s2)):
      if s2[i] == ')' and s2[i - dp[i - 1] - 1] == '(':
        dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2

    return max(dp)


# Link: https://leetcode.com/problems/split-array-largest-sum/description/
class Solution:
  def splitArray(self, nums: List[int], k: int) -> int:
    prefix = list(itertools.accumulate(nums, initial=0))

    @functools.lru_cache(None)
    def dp(i: int, k: int) -> int:
      """
      Returns the minimum of the maximum sum to split the first i numbers into
      k groups.
      """
      if k == 1:
        return prefix[i]
      return min(max(dp(j, k - 1), prefix[i] - prefix[j])
                 for j in range(k - 1, i))

    return dp(len(nums), k)


# Link: https://leetcode.com/problems/split-array-largest-sum/description/
class Solution:
  def splitArray(self, nums: List[int], k: int) -> int:
    n = len(nums)
    # dp[i][k] := the minimum of the maximum sum to split the first i numbers
    # into k groups
    dp = [[math.inf] * (k + 1) for _ in range(n + 1)]
    prefix = [0] + list(itertools.accumulate(nums))

    for i in range(1, n + 1):
      dp[i][1] = prefix[i]

    for l in range(2, k + 1):
      for i in range(l, n + 1):
        for j in range(l - 1, i):
          dp[i][l] = min(dp[i][l], max(dp[j][l - 1], prefix[i] - prefix[j]))

    return dp[n][k]


# Link: https://leetcode.com/problems/split-array-largest-sum/description/
class Solution:
  def splitArray(self, nums: List[int], k: int) -> int:
    l = max(nums)
    r = sum(nums) + 1

    def numGroups(maxSumInGroup: int) -> int:
      groupCount = 1
      sumInGroup = 0

      for num in nums:
        if sumInGroup + num <= maxSumInGroup:
          sumInGroup += num
        else:
          groupCount += 1
          sumInGroup = num

      return groupCount

    while l < r:
      m = (l + r) // 2
      if numGroups(m) > k:
        l = m + 1
      else:
        r = m

    return l


# Link: https://leetcode.com/problems/freedom-trail/description/
class Solution:
  def findRotateSteps(self, ring: str, key: str) -> int:
    @functools.lru_cache(None)
    def dfs(ring: str, index: int) -> int:
      """Returns the number of rotates of ring to match key[index..n)."""
      if index == len(key):
        return 0

      ans = math.inf

      # For each ring[i] == key[index], we rotate the ring to match the ring[i]
      # with the key[index], then recursively match the newRing with the
      # key[index + 1..n).
      for i, r in enumerate(ring):
        if r == key[index]:
          minRotates = min(i, len(ring) - i)
          newRing = ring[i:] + ring[:i]
          remainingRotates = dfs(newRing, index + 1)
          ans = min(ans, minRotates + remainingRotates)

      return ans

    return dfs(ring, 0) + len(key)


# Link: https://leetcode.com/problems/closest-room/description/
from sortedcontainers import SortedList


class Solution:
  def closestRoom(self, rooms: List[List[int]], queries: List[List[int]]) -> List[int]:
    ans = [0] * len(queries)
    qs = [[*q, i] for i, q in enumerate(queries)]
    roomIds = SortedList()

    rooms.sort(key=lambda x: -x[1])
    qs.sort(key=lambda x: -x[1])

    def searchClosestRoomId(roomIds: SortedList, preferred: int):
      if not roomIds:
        return -1

      candIds = []
      i = roomIds.bisect_right(preferred)
      if i > 0:
        candIds.append(roomIds[i - 1])
      if i < len(roomIds):
        candIds.append(roomIds[i])
      return min(candIds, key=lambda x: abs(x - preferred))

    i = 0  # rooms' index
    for preferred, minSize, index in qs:
      while i < len(rooms) and rooms[i][1] >= minSize:
        roomIds.add(rooms[i][0])
        i += 1
      ans[index] = searchClosestRoomId(roomIds, preferred)

    return ans


# Link: https://leetcode.com/problems/reachable-nodes-in-subdivided-graph/description/
class Solution:
  def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
    graph = [[] for _ in range(n)]
    dist = [maxMoves + 1] * n

    for u, v, cnt in edges:
      graph[u].append((v, cnt))
      graph[v].append((u, cnt))

    reachableNodes = self._dijkstra(graph, 0, maxMoves, dist)
    reachableSubnodes = 0

    for u, v, cnt in edges:
      # the number of reachable nodes of (u, v) from `u`
      a = 0 if dist[u] > maxMoves else min(maxMoves - dist[u], cnt)
      # the number of reachable nodes of (u, v) from `v`
      b = 0 if dist[v] > maxMoves else min(maxMoves - dist[v], cnt)
      reachableSubnodes += min(a + b, cnt)

    return reachableNodes + reachableSubnodes

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, maxMoves: int, dist: List[int]) -> int:
    dist[src] = 0
    minHeap = [(dist[src], src)]  # (d, u)

    while minHeap:
      d, u = heapq.heappop(minHeap)
      # Already took `maxMoves` to reach `u`, so can't explore anymore.
      if dist[u] >= maxMoves:
        break
      for v, w in graph[u]:
        newDist = d + w + 1
        if newDist < dist[v]:
          dist[v] = newDist
          heapq.heappush(minHeap, (newDist, v))

    return sum(d <= maxMoves for d in dist)


# Link: https://leetcode.com/problems/find-all-people-with-secret/description/
class UnionFind:
  def __init__(self, n: int):
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

  def connected(self, u: int, v: int) -> bool:
    return self._find(self.id[u]) == self._find(self.id[v])

  def reset(self, u: int) -> None:
    self.id[u] = u

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
    uf = UnionFind(n)
    timeToPairs = collections.defaultdict(list)

    uf.unionByRank(0, firstPerson)

    for x, y, time in meetings:
      timeToPairs[time].append((x, y))

    for _, pairs in sorted(timeToPairs.items(), key=lambda x: x[0]):
      peopleUnioned = set()
      for x, y in pairs:
        uf.unionByRank(x, y)
        peopleUnioned.add(x)
        peopleUnioned.add(y)
      for person in peopleUnioned:
        if not uf.connected(person, 0):
          uf.reset(person)

    return [i for i in range(n) if uf.connected(i, 0)]


# Link: https://leetcode.com/problems/earliest-possible-day-of-full-bloom/description/
class Solution:
  def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
    ans = 0
    time = 0

    for p, g in sorted([(p, g) for (p, g) in zip(plantTime, growTime)], key=lambda x: -x[1]):
      time += p
      ans = max(ans, time + g)

    return ans


# Link: https://leetcode.com/problems/alien-dictionary/description/
class Solution:
  def alienOrder(self, words: List[str]) -> str:
    graph = {}
    inDegrees = [0] * 26

    self._buildGraph(graph, words, inDegrees)
    return self._topology(graph, inDegrees)

  def _buildGraph(self, graph: Dict[chr, Set[chr]], words: List[str], inDegrees: List[int]) -> None:
    # Create a node for each character in each word.
    for word in words:
      for c in word:
        if c not in graph:
          graph[c] = set()

    for first, second in zip(words, words[1:]):
      length = min(len(first), len(second))
      for j in range(length):
        u = first[j]
        v = second[j]
        if u != v:
          if v not in graph[u]:
            graph[u].add(v)
            inDegrees[ord(v) - ord('a')] += 1
          break  # The order of characters after this are meaningless.
        # First = 'ab', second = 'a' . invalid
        if j == length - 1 and len(first) > len(second):
          graph.clear()
          return

  def _topology(self, graph: Dict[chr, Set[chr]], inDegrees: List[int]) -> str:
    s = ''
    q = collections.deque()

    for c in graph:
      if inDegrees[ord(c) - ord('a')] == 0:
        q.append(c)

    while q:
      u = q.pop()
      s += u
      for v in graph[u]:
        inDegrees[ord(v) - ord('a')] -= 1
        if inDegrees[ord(v) - ord('a')] == 0:
          q.append(v)

    # Words = ['z', 'x', 'y', 'x']
    return s if len(s) == len(graph) else ''


# Link: https://leetcode.com/problems/maximum-deletions-on-a-string/description/
class Solution:
  def deleteString(self, s: str) -> int:
    n = len(s)
    # lcs[i][j] := the number of the same letters of s[i..n) and s[j..n)
    lcs = [[0] * (n + 1) for _ in range(n + 1)]
    # dp[i] := the maximum number of operations needed to delete s[i..n)
    dp = [1] * n

    for i in reversed(range(n)):
      for j in range(i + 1, n):
        if s[i] == s[j]:
          lcs[i][j] = lcs[i + 1][j + 1] + 1
        if lcs[i][j] >= j - i:
          dp[i] = max(dp[i], dp[j] + 1)

    return dp[0]


# Link: https://leetcode.com/problems/nth-magical-number/description/
class Solution:
  def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
    lcm = a * b // math.gcd(a, b)
    l = bisect.bisect_left(range(min(a, b), min(a, b) * n), n,
                           key=lambda m: m // a + m // b - m // lcm) + min(a, b)
    return l % (10**9 + 7)


# Link: https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/description/
class Solution:
  def ways(self, pizza: List[str], k: int) -> int:
    kMod = 1_000_000_007
    M = len(pizza)
    N = len(pizza[0])
    prefix = [[0] * (N + 1) for _ in range(M + 1)]

    for i in range(M):
      for j in range(N):
        prefix[i + 1][j + 1] = (pizza[i][j] == 'A') + \
            prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j]

    def hasApple(row1: int, row2: int, col1: int, col2: int) -> bool:
      """Returns True if pizza[row1..row2)[col1..col2) has apple."""
      return (prefix[row2][col2] - prefix[row1][col2] -
              prefix[row2][col1] + prefix[row1][col1]) > 0

    @functools.lru_cache(None)
    def dp(m: int, n: int, k: int) -> int:
      """Returns the number of ways to cut pizza[m..M)[n..N) with k cuts."""
      if k == 0:
        return 1 if hasApple(m, M, n, N) else 0

      res = 0

      for i in range(m + 1, M):  # Cut horizontally.
        if hasApple(m, i, n, N) and hasApple(i, M, n, N):
          res += dp(i, n, k - 1)

      for j in range(n + 1, N):  # Cut vertically.
        if hasApple(m, M, n, j) and hasApple(m, M, j, N):
          res += dp(m, j, k - 1)

      return res % kMod

    return dp(0, 0, k - 1)


# Link: https://leetcode.com/problems/candy/description/
class Solution:
  def candy(self, ratings: List[int]) -> int:
    n = len(ratings)

    ans = 0
    l = [1] * n
    r = [1] * n

    for i in range(1, n):
      if ratings[i] > ratings[i - 1]:
        l[i] = l[i - 1] + 1

    for i in range(n - 2, -1, -1):
      if ratings[i] > ratings[i + 1]:
        r[i] = r[i + 1] + 1

    for a, b in zip(l, r):
      ans += max(a, b)

    return ans


# Link: https://leetcode.com/problems/minimum-time-to-complete-all-tasks/description/
class Solution:
  def findMinimumTime(self, tasks: List[List[int]]) -> int:
    kMax = 2000
    running = [False] * (kMax + 1)

    # Sort tasks by end.
    for start, end, duration in sorted(tasks, key=lambda x: x[1]):
      neededDuration = duration - \
          sum(running[i] for i in range(start, end + 1))
      # Greedily run the task as late as possible so that later tasks can run
      # simultaneously.
      i = end
      while neededDuration > 0:
        if not running[i]:
          running[i] = True
          neededDuration -= 1
        i -= 1

    return sum(running)


# Link: https://leetcode.com/problems/regular-expression-matching/description/
class Solution:
  def isMatch(self, s: str, p: str) -> bool:
    m = len(s)
    n = len(p)
    # dp[i][j] := True if s[0..i) matches p[0..j)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    def isMatch(i: int, j: int) -> bool:
      return j >= 0 and p[j] == '.' or s[i] == p[j]

    for j, c in enumerate(p):
      if c == '*' and dp[0][j - 1]:
        dp[0][j + 1] = True

    for i in range(m):
      for j in range(n):
        if p[j] == '*':
          # The minimum index of '*' is 1.
          noRepeat = dp[i + 1][j - 1]
          doRepeat = isMatch(i, j - 1) and dp[i][j + 1]
          dp[i + 1][j + 1] = noRepeat or doRepeat
        elif isMatch(i, j):
          dp[i + 1][j + 1] = dp[i][j]

    return dp[m][n]


# Link: https://leetcode.com/problems/maximum-segment-sum-after-removals/description/
class Solution:
  def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
    n = len(nums)
    maxSum = 0
    ans = [0] * n
    # For the segment [l, r], record its sum in summ[l] and summ[r]
    summ = [0] * n
    # For the segment [l, r], record its count in count[l] and count[r]
    count = [0] * n

    for i in reversed(range(n)):
      ans[i] = maxSum
      j = removeQueries[i]

      # Calculate `segmentSum`.
      leftSum = summ[j - 1] if j > 0 else 0
      rightSum = summ[j + 1] if j + 1 < n else 0
      segmentSum = nums[j] + leftSum + rightSum

      # Calculate `segmentCount`.
      leftCount = count[j - 1] if j > 0 else 0
      rightCount = count[j + 1] if j + 1 < n else 0
      segmentCount = 1 + leftCount + rightCount

      # Update `summ` and `count` of the segment [l, r].
      l = j - leftCount
      r = j + rightCount
      summ[l] = segmentSum
      summ[r] = segmentSum
      count[l] = segmentCount
      count[r] = segmentCount
      maxSum = max(maxSum, segmentSum)

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits-ii/description/
class Solution:
  # Same as 2944. Minimum Number of Coins for Fruits
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


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits-ii/description/
class Solution:
  # Same as 2944. Minimum Number of Coins for Fruits
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


# Link: https://leetcode.com/problems/minimum-number-of-coins-for-fruits-ii/description/
class Solution:
  # Same as 2944. Minimum Number of Coins for Fruits
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


# Link: https://leetcode.com/problems/chalkboard-xor-game/description/
class Solution:
  def xorGame(self, nums: List[int]) -> bool:
    return functools.reduce(operator.xor, nums) == 0 or len(nums) % 2 == 0


# Link: https://leetcode.com/problems/minimum-costs-using-the-train-line/description/
class Solution:
  def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
    n = len(regular)
    ans = [0] * n
    # the minimum cost to reach the current stop in a regular route
    dpReg = 0
    # the minimum cost to reach the current stop in an express route
    dpExp = expressCost

    for i in range(n):
      prevReg = dpReg
      prevExp = dpExp
      dpReg = min(prevReg + regular[i], prevExp + 0 + regular[i])
      dpExp = min(prevReg + expressCost + express[i], prevExp + express[i])
      ans[i] = min(dpReg, dpExp)

    return ans


# Link: https://leetcode.com/problems/special-binary-string/description/
class Solution:
  def makeLargestSpecial(self, s: str) -> str:
    specials = []
    count = 0

    i = 0
    for j, c in enumerate(s):
      count += 1 if c == '1' else -1
      if count == 0:
        specials.append(
            '1' + self.makeLargestSpecial(s[i + 1:j]) + '0')
        i = j + 1

    return ''.join(sorted(specials)[::-1])


# Link: https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/description/
class Solution:
  def minimumDifference(self, nums: List[int]) -> int:
    n = len(nums) // 2
    summ = sum(nums)
    goal = summ // 2
    lNums = nums[:n]
    rNums = nums[n:]
    ans = abs(sum(lNums) - sum(rNums))
    lSums = [[] for _ in range(n + 1)]
    rSums = [[] for _ in range(n + 1)]

    def dfs(A: List[int], i: int, count: int, path: int, sums: List[List[int]]):
      if i == len(A):
        sums[count].append(path)
        return
      dfs(A, i + 1, count + 1, path + A[i], sums)
      dfs(A, i + 1, count, path, sums)

    dfs(lNums, 0, 0, 0, lSums)
    dfs(rNums, 0, 0, 0, rSums)

    for lCount in range(n):
      l = lSums[lCount]
      r = rSums[n - lCount]
      r.sort()
      for lSum in l:
        i = bisect_left(r, goal - lSum)
        if i < len(r):
          sumPartOne = summ - lSum - r[i]
          sumPartTwo = summ - sumPartOne
          ans = min(ans, abs(sumPartOne - sumPartTwo))
        if i > 0:
          sumPartOne = summ - lSum - r[i - 1]
          sumPartTwo = summ - sumPartOne
          ans = min(ans, abs(sumPartOne - sumPartTwo))

    return ans


# Link: https://leetcode.com/problems/best-position-for-a-service-centre/description/
class Solution:
  def getMinDistSum(self, positions: List[List[int]]) -> float:
    def distSum(a: float, b: float) -> float:
      return sum(math.sqrt((a - x)**2 + (b - y)**2)
                 for x, y in positions)

    kErr = 1e-6
    currX = 50
    currY = 50
    ans = distSum(currX, currY)
    step = 1

    while step > kErr:
      shouldDecreaseStep = True
      for dx, dy in [(0, step), (0, -step), (step, 0), (-step, 0)]:
        x = currX + dx
        y = currY + dy
        newDistSum = distSum(x, y)
        if newDistSum < ans:
          ans = newDistSum
          currX = x
          currY = y
          shouldDecreaseStep = False
      if shouldDecreaseStep:
        step /= 10

    return ans


# Link: https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/description/
class Solution:
  def countPaths(self, grid: List[List[int]]) -> int:
    kMod = 1_000_000_007
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])

    @functools.lru_cache(None)
    def dp(i: int, j: int) -> int:
      """Returns the number of increasing paths starting from (i, j)."""
      ans = 1  # The current cell contributes 1 length.
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if grid[x][y] <= grid[i][j]:
          continue
        ans += dp(x, y)
        ans %= kMod
      return ans

    return sum(dp(i, j)
               for i in range(m)
               for j in range(n)) % kMod


# Link: https://leetcode.com/problems/rearranging-fruits/description/
class Solution:
  def minCost(self, basket1: List[int], basket2: List[int]) -> int:
    swapped = []
    count = collections.Counter(basket1)
    count.subtract(collections.Counter(basket2))
    
    for num, freq in count.items():
      if freq % 2 != 0:
        return -1
      swapped += [num] * abs(freq // 2)
    
    swapped.sort()
    minNum = min(min(basket1), min(basket2))
    # Other than directly swap basket1[i] and basket2[j], we can swap basket1[i]
    # with `minNum` first then swap `minNum` with basket2[j], and vice versa.
    # That's why we take min(2 * minNum, num) in the below.
    return sum(min(2 * minNum, num) for num in swapped[0:len(swapped) // 2])
  

# Link: https://leetcode.com/problems/number-of-ways-to-separate-numbers/description/
class Solution:
  def numberOfCombinations(self, num: str) -> int:
    if num[0] == '0':
      return 0

    kMod = 1_000_000_007
    n = len(num)
    # dp[i][k] := the number of possible lists of integers ending in num[i]
    # with the length of the last number being 1..k
    dp = [[0] * (n + 1) for _ in range(n)]
    # lcs[i][j] := the number of the same digits in num[i..n) and num[j..n)
    lcs = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n - 1, -1, -1):
      for j in range(i + 1, n):
        if num[i] == num[j]:
          lcs[i][j] = lcs[i + 1][j + 1] + 1

    for i in range(n):
      for k in range(1, i + 2):
        dp[i][k] += dp[i][k - 1]
        dp[i][k] %= kMod
        # The last number is num[s..i].
        s = i - k + 1
        if num[s] == '0':
          # the number of possible lists of integers ending in num[i] with the
          # length of the last number being k
          continue
        if s == 0:  # the whole string
          dp[i][k] += 1
          continue
        if s < k:
          # The length k is not enough, so add the number of possible lists of
          # integers in num[0..s - 1].
          dp[i][k] += dp[s - 1][s]
          continue
        l = lcs[s - k][s]
        if l >= k or num[s - k + l] <= num[s + l]:
          # Have enough length k and num[s - k..s - 1] <= num[j..i].
          dp[i][k] += dp[s - 1][k]
        else:
          # Have enough length k but num[s - k..s - 1] > num[j..i].
          dp[i][k] += dp[s - 1][k - 1]

    return dp[n - 1][n] % kMod


# Link: https://leetcode.com/problems/maximum-number-of-k-divisible-components/description/
class Solution:
  def maxKDivisibleComponents(self, n: int, edges: List[List[int]], values: List[int], k: int) -> int:
    ans = 0
    graph = [[] for _ in range(n)]

    def dfs(u: int, prev: int) -> int:
      nonlocal ans
      treeSum = values[u]

      for v in graph[u]:
        if v != prev:
          treeSum += dfs(v, u)

      if treeSum % k == 0:
        ans += 1
      return treeSum

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    dfs(0, -1)
    return ans


# Link: https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/description/
class UnionFind:
  def __init__(self, n: int):
    self.count = n
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
    self.count -= 1
    return True

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def maxNumEdgesToRemove(self, n: int, edges: List[List[int]]) -> int:
    alice = UnionFind(n)
    bob = UnionFind(n)
    requiredEdges = 0

    # Greedily put type 3 edges in the front.
    for type, u, v in sorted(edges, reverse=True):
      u -= 1
      v -= 1
      if type == 3:  # Can be traversed by Alice and Bob.
          # Note that we should use | instead of or because if the first
          # expression is True, short-circuiting will skip the second
          # expression.
        if alice.unionByRank(u, v) | bob.unionByRank(u, v):
          requiredEdges += 1
      elif type == 2:  # Can be traversed by Bob.
        if bob.unionByRank(u, v):
          requiredEdges += 1
      else:  # type == 1 Can be traversed by Alice.
        if alice.unionByRank(u, v):
          requiredEdges += 1

    return len(edges) - requiredEdges \
        if alice.count == 1 and bob.count == 1 \
        else -1


# Link: https://leetcode.com/problems/consecutive-numbers-sum/description/
class Solution:
  def consecutiveNumbersSum(self, n: int) -> int:
    ans = 0
    i = 1
    triangleNum = 1
    while triangleNum <= n:
      if (n - triangleNum) % i == 0:
        ans += 1
      i += 1
      triangleNum += i
    return ans


# Link: https://leetcode.com/problems/stone-game-viii/description/
class Solution:
  def stoneGameVIII(self, stones: List[int]) -> int:
    n = len(stones)
    prefix = list(itertools.accumulate(stones))
    # dp[i] := the maximum score difference the current player can get when the
    # game starts at i, i.e. stones[0..i] are merged into the value prefix[i]
    dp = [-math.inf] * n

    # Must take all when there're only two stones left.
    dp[n - 2] = prefix[-1]

    for i in reversed(range(n - 2)):
      dp[i] = max(dp[i + 1], prefix[i + 1] - dp[i + 1])

    return dp[0]


# Link: https://leetcode.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/description/
class Solution:
  def smallestSubsequence(self, s: str, k: int, letter: str, repetition: int) -> str:
    stack = []  # running string
    required = repetition
    nLetters = s.count(letter)

    for i, c in enumerate(s):
      # Make sure the length is sufficient:
      # Len(stack) := the length of running string
      # Len(s) - i := the length of remain chars
      # -1 := we're going to pop a char
      while stack and stack[-1] > c \
              and len(stack) + len(s) - i - 1 >= k \
              and (stack[-1] != letter or nLetters > required):
        if stack.pop() == letter:
          required += 1
      if len(stack) < k:
        if c == letter:
          stack.append(c)
          required -= 1
        elif k - len(stack) > required:
          stack.append(c)
      if c == letter:
        nLetters -= 1

    return ''.join(stack)


# Link: https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/description/
class Solution:
  def minimumTime(self, s: str) -> int:
    n = len(s)
    # left[i] := the minimum time to remove the illegal cars of s[0..i]
    left = [0] * n
    left[0] = ord(s[0]) - ord('0')
    # dp[i] := the minimum time to remove the illegal cars of s[0..i] optimally
    # + the time to remove the illegal cars of s[i + 1..n) consecutively
    # Note that the way to remove the illegal cars in the right part
    # doesn't need to be optimal since:
    #   `left | illegal cars | n - 1 - k` will be covered in
    #   `left' | n - 1 - i` later.
    dp = [n] * n
    dp[0] = left[0] + n - 1

    for i in range(1, n):
      left[i] = min(left[i - 1] + (ord(s[i]) - ord('0')) * 2, i + 1)
      dp[i] = min(dp[i], left[i] + n - 1 - i)

    return min(dp)


# Link: https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/description/
class Solution:
  def minimumTime(self, s: str) -> int:
    n = len(s)
    ans = n
    left = 0  # the minimum time to remove the illegal cars so far

    for i, c in enumerate(s):
      left = min(left + (ord(c) - ord('0')) * 2, i + 1)
      ans = min(ans, left + n - 1 - i)

    return ans


# Link: https://leetcode.com/problems/brace-expansion-ii/description/
class Solution:
  def braceExpansionII(self, expression: str) -> List[str]:
    def merge(groups: List[List[str]], group: List[str]) -> None:
      if not groups[-1]:
        groups[-1] = group
        return

      groups[-1] = [word1 + word2 for word1 in groups[-1]
                    for word2 in group]

    def dfs(s: int, e: int) -> List[str]:
      groups = [[]]
      layer = 0

      for i in range(s, e + 1):
        c = expression[i]
        if c == '{':
          layer += 1
          if layer == 1:
            left = i + 1
        elif c == '}':
          layer -= 1
          if layer == 0:
            group = dfs(left, i - 1)
            merge(groups, group)
        elif c == ',' and layer == 0:
          groups.append([])
        elif layer == 0:
          merge(groups, [c])

      return sorted(list({word for group in groups for word in group}))

    return dfs(0, len(expression) - 1)


# Link: https://leetcode.com/problems/count-of-range-sum/description/
class Solution:
  def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
    n = len(nums)
    self.ans = 0
    prefix = [0] + list(itertools.accumulate(nums))

    self._mergeSort(prefix, 0, n, lower, upper)
    return self.ans

  def _mergeSort(self, prefix: List[int], l: int, r: int, lower: int, upper: int) -> None:
    if l >= r:
      return

    m = (l + r) // 2
    self._mergeSort(prefix, l, m, lower, upper)
    self._mergeSort(prefix, m + 1, r, lower, upper)
    self._merge(prefix, l, m, r, lower, upper)

  def _merge(self, prefix: List[int], l: int, m: int, r: int, lower: int, upper: int) -> None:
    lo = m + 1  # the first index s.t. prefix[lo] - prefix[i] >= lower
    hi = m + 1  # the first index s.t. prefix[hi] - prefix[i] > upper

    # For each index i in range [l, m], add hi - lo to `ans`.
    for i in range(l, m + 1):
      while lo <= r and prefix[lo] - prefix[i] < lower:
        lo += 1
      while hi <= r and prefix[hi] - prefix[i] <= upper:
        hi += 1
      self.ans += hi - lo

    sorted = [0] * (r - l + 1)
    k = 0      # sorted's index
    i = l      # left's index
    j = m + 1  # right's index

    while i <= m and j <= r:
      if prefix[i] < prefix[j]:
        sorted[k] = prefix[i]
        k += 1
        i += 1
      else:
        sorted[k] = prefix[j]
        k += 1
        j += 1

    # Put the possible remaining left part into the sorted array.
    while i <= m:
      sorted[k] = prefix[i]
      k += 1
      i += 1

    # Put the possible remaining right part into the sorted array.
    while j <= r:
      sorted[k] = prefix[j]
      k += 1
      j += 1

    prefix[l:l + len(sorted)] = sorted


# Link: https://leetcode.com/problems/k-inverse-pairs-array/description/
class Solution:
  def kInversePairs(self, n: int, k: int) -> int:
    kMod = 1_000_000_007
    # dp[i][j] := the number of permutations of numbers 1..i with j inverse pairs
    dp = [[0] * (k + 1) for _ in range(n + 1)]

    # If there's no inverse pair, the permutation is unique '123..i'
    for i in range(n + 1):
      dp[i][0] = 1

    for i in range(1, n + 1):
      for j in range(1, k + 1):
        dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % kMod
        if j - i >= 0:
          dp[i][j] = (dp[i][j] - dp[i - 1][j - i] + kMod) % kMod

    return dp[n][k]


# Link: https://leetcode.com/problems/max-points-on-a-line/description/
class Solution:
  def maxPoints(self, points: List[List[int]]) -> int:
    ans = 0

    def gcd(a: int, b: int) -> int:
      return a if b == 0 else gcd(b, a % b)

    def getSlope(p: List[int], q: List[int]) -> Tuple[int, int]:
      dx = p[0] - q[0]
      dy = p[1] - q[1]
      if dx == 0:
        return (0, p[0])
      if dy == 0:
        return (p[1], 0)
      d = gcd(dx, dy)
      return (dx // d, dy // d)

    for i, p in enumerate(points):
      slopeCount = collections.defaultdict(int)
      samePoints = 1
      maxPoints = 0  # the maximum number of points with the same slope
      for j in range(i + 1, len(points)):
        q = points[j]
        if p == q:
          samePoints += 1
        else:
          slope = getSlope(p, q)
          slopeCount[slope] += 1
          maxPoints = max(maxPoints, slopeCount[slope])
      ans = max(ans, samePoints + maxPoints)

    return ans


# Link: https://leetcode.com/problems/build-a-matrix-with-conditions/description/
class Solution:
  def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
    rowOrder = self._topologicalSort(rowConditions, k)
    if not rowOrder:
      return []

    colOrder = self._topologicalSort(colConditions, k)
    if not colOrder:
      return []

    ans = [[0] * k for _ in range(k)]
    nodeToRowIndex = [0] * (k + 1)

    for i, node in enumerate(rowOrder):
      nodeToRowIndex[node] = i

    for j, node in enumerate(colOrder):
      i = nodeToRowIndex[node]
      ans[i][j] = node

    return ans

  def _topologicalSort(self, conditions: List[List[int]], n: int) -> List[int]:
    order = []
    graph = [[] for _ in range(n + 1)]
    inDegrees = [0] * (n + 1)

    # Build the graph.
    for u, v in conditions:
      graph[u].append(v)
      inDegrees[v] += 1

    # Perform topological sorting.
    q = collections.deque([i for i in range(1, n + 1) if inDegrees[i] == 0])

    while q:
      u = q.popleft()
      order.append(u)
      for v in graph[u]:
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return order if len(order) == n else []


# Link: https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
class Solution:
  def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
    ans = math.inf
    qualitySum = 0
    # (wagePerQuality, quality) sorted by wagePerQuality
    workers = sorted((w / q, q) for q, w in zip(quality, wage))
    maxHeap = []

    for wagePerQuality, q in workers:
      heapq.heappush(maxHeap, -q)
      qualitySum += q
      if len(maxHeap) > k:
        qualitySum += heapq.heappop(maxHeap)
      if len(maxHeap) == k:
        ans = min(ans, qualitySum * wagePerQuality)

    return ans


# Link: https://leetcode.com/problems/second-minimum-time-to-reach-destination/description/
class Solution:
  def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
    graph = [[] for _ in range(n + 1)]
    q = collections.deque([(1, 0)])
    # minTime[i][0] := the first minimum time to reach the node i
    # minTime[i][1] := the second minimum time to reach the node i
    minTime = [[math.inf] * 2 for _ in range(n + 1)]
    minTime[1][0] = 0

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    while q:
      i, prevTime = q.popleft()
      # Start from green.
      # If `numChangeSignal` is odd, now red.
      # If numChangeSignal is even -> now gree
      numChangeSignal = prevTime // change
      waitTime = change - (prevTime % change) if numChangeSignal & 1 else 0
      newTime = prevTime + waitTime + time
      for j in graph[i]:
        if newTime < minTime[j][0]:
          minTime[j][0] = newTime
          q.append((j, newTime))
        elif minTime[j][0] < newTime < minTime[j][1]:
          if j == n:
            return newTime
          minTime[j][1] = newTime
          q.append((j, newTime))


# Link: https://leetcode.com/problems/greatest-common-divisor-traversal/description/
class UnionFind:
  def __init__(self, n: int):
    self.id = list(range(n))
    self.sz = [1] * n

  def unionBySize(self, u: int, v: int) -> None:
    i = self._find(u)
    j = self._find(v)
    if i == j:
      return
    if self.sz[i] < self.sz[j]:
      self.sz[j] += self.sz[i]
      self.id[i] = j
    else:
      self.sz[i] += self.sz[j]
      self.id[j] = i

  def getSize(self, i: int) -> int:
    return self.sz[i]

  def _find(self, u: int) -> int:
    if self.id[u] != u:
      self.id[u] = self._find(self.id[u])
    return self.id[u]


class Solution:
  def canTraverseAllPairs(self, nums: List[int]) -> bool:
    n = len(nums)
    max_num = max(nums)
    maxPrimeFactor = self._sieveEratosthenes(max_num + 1)
    primeToFirstIndex = collections.defaultdict(int)
    uf = UnionFind(n)

    for i, num in enumerate(nums):
      for prime_factor in self._getPrimeFactors(num, maxPrimeFactor):
        if prime_factor in primeToFirstIndex:
          uf.unionBySize(primeToFirstIndex[prime_factor], i)
        else:
          primeToFirstIndex[prime_factor] = i

    return any(uf.getSize(i) == n for i in range(n))

  def _sieveEratosthenes(self, n: int) -> List[int]:
    """Gets the minimum prime factor of i, where 1 < i <= n."""
    minPrimeFactors = [i for i in range(n + 1)]
    for i in range(2, int(n**0.5) + 1):
      if minPrimeFactors[i] == i:  # `i` is prime.
        for j in range(i * i, n, i):
          minPrimeFactors[j] = min(minPrimeFactors[j], i)
    return minPrimeFactors

  def _getPrimeFactors(self, num: int, minPrimeFactors: List[int]) -> List[int]:
    primeFactors = []
    while num > 1:
      divisor = minPrimeFactors[num]
      primeFactors.append(divisor)
      while num % divisor == 0:
        num //= divisor
    return primeFactors


# Link: https://leetcode.com/problems/first-missing-positive/description/
class Solution:
  def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)

    # Correct slot:
    # nums[i] = i + 1
    # nums[i] - 1 = i
    # nums[nums[i] - 1] = nums[i]
    for i in range(n):
      while nums[i] > 0 and nums[i] <= n and nums[nums[i] - 1] != nums[i]:
        nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    for i, num in enumerate(nums):
      if num != i + 1:
        return i + 1

    return n + 1


# Link: https://leetcode.com/problems/number-of-distinct-roll-sequences/description/
class Solution:
  def distinctSequences(self, n: int) -> int:
    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(n: int, prev: int, prevPrev: int) -> int:
      """
      Returns the number of distinct sequences for n dices with `prev` and
      `prevPrev`.
      """
      if n == 0:
        return 1
      res = 0
      for dice in range(1, 7):
        if dice != prev and dice != prevPrev and \
                (prev == 0 or math.gcd(dice, prev) == 1):
          res += dp(n - 1, dice, prev)
          res %= kMod
      return res

    return dp(n, 0, 0)


# Link: https://leetcode.com/problems/the-most-similar-path-in-a-graph/description/
class Solution:
  def mostSimilar(self, n: int, roads: List[List[int]], names: List[str],
                  targetPath: List[str]) -> List[int]:
    # cost[i][j] := the minimum cost to start from names[i] in path[j]
    cost = [[-1] * len(targetPath) for _ in range(len(names))]
    # next[i][j] := the best next of names[i] in path[j]
    next = [[0] * len(targetPath) for _ in range(len(names))]
    graph = [[] for _ in range(n)]

    for u, v in roads:
      graph[u].append(v)
      graph[v].append(u)

    minDist = math.inf
    start = 0

    def dfs(nameIndex: int, pathIndex: int) -> int:
      if cost[nameIndex][pathIndex] != -1:
        return cost[nameIndex][pathIndex]

      editDist = names[nameIndex] != targetPath[pathIndex]
      if pathIndex == len(targetPath) - 1:
        return editDist

      minDist = math.inf

      for v in graph[nameIndex]:
        dist = dfs(v, pathIndex + 1)
        if dist < minDist:
          minDist = dist
          next[nameIndex][pathIndex] = v

      cost[nameIndex][pathIndex] = editDist + minDist
      return editDist + minDist

    for i in range(n):
      dist = dfs(i, 0)
      if dist < minDist:
        minDist = dist
        start = i

    ans = []

    while len(ans) < len(targetPath):
      ans.append(start)
      start = next[start][len(ans) - 1]

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/description/
class Solution:
  def maxPalindromes(self, s: str, k: int) -> int:
    n = len(s)
    # dp[i] := the maximum number of substrings in the first i chars of s
    dp = [0] * (n + 1)

    def isPalindrome(l: int, r: int) -> bool:
      """Returns True is s[i..j) is a palindrome."""
      if l < 0:
        return False
      while l < r:
        if s[l] != s[r]:
          return False
        l += 1
        r -= 1
      return True

    # If a palindrome is a subof another palindrome, then considering
    # the longer palindrome won't increase the number of non-overlapping
    # palindromes. So, we only need to consider the shorter one. Also,
    # considering palindromes with both k length and k + 1 length ensures that
    # we look for both even and odd length palindromes.
    for i in range(k, n + 1):
      dp[i] = dp[i - 1]
      # Consider palindrome with length k.
      if isPalindrome(i - k, i - 1):
        dp[i] = max(dp[i], 1 + dp[i - k])
      # Consider palindrome with length k + 1.
      if isPalindrome(i - k - 1, i - 1):
        dp[i] = max(dp[i], 1 + dp[i - k - 1])

    return dp[n]


# Link: https://leetcode.com/problems/count-nodes-that-are-great-enough/description/
class Solution:
  def countGreatEnoughNodes(self, root: Optional[TreeNode], k: int) -> int:
    ans = 0

    def dfs(root: Optional[TreeNode]) -> List[int]:
      nonlocal ans
      if not root:
        return []

      kSmallest = sorted(dfs(root.left) + dfs(root.right))[:k]
      if len(kSmallest) == k and root.val > kSmallest[-1]:
        ans += 1

      return kSmallest + [root.val]

    dfs(root)
    return ans


# Link: https://leetcode.com/problems/count-valid-paths-in-a-tree/description/
class Solution:
  def countPaths(self, n: int, edges: List[List[int]]) -> int:
    ans = 0
    isPrime = self._sieveEratosthenes(n + 1)
    graph = [[] for _ in range(n + 1)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    def dfs(u: int, prev: int) -> Tuple[int, int]:
      nonlocal ans
      countZeroPrimePath = int(not isPrime[u])
      countOnePrimePath = int(isPrime[u])

      for v in graph[u]:
        if v == prev:
          continue
        countZeroPrimeChildPath, countOnePrimeChildPath = dfs(v, u)
        ans += countZeroPrimePath * countOnePrimeChildPath + \
            countOnePrimePath * countZeroPrimeChildPath
        if isPrime[u]:
          countOnePrimePath += countZeroPrimeChildPath
        else:
          countZeroPrimePath += countZeroPrimeChildPath
          countOnePrimePath += countOnePrimeChildPath

      return countZeroPrimePath, countOnePrimePath

    dfs(1, -1)
    return ans

  def _sieveEratosthenes(self, n: int) -> List[bool]:
    isPrime = [True] * n
    isPrime[0] = False
    isPrime[1] = False
    for i in range(2, int(n**0.5) + 1):
      if isPrime[i]:
        for j in range(i * i, n, i):
          isPrime[j] = False
    return isPrime


# Link: https://leetcode.com/problems/maximum-element-sum-of-a-complete-subset-of-indices/description/
class Solution:
  def maximumSum(self, nums: List[int]) -> int:
    ans = 0

    for oddPower in range(1, len(nums) + 1):
      summ = 0
      for num in range(1, len(nums) + 1):
        if num * num * oddPower > len(nums):
          break
        summ += nums[oddPower * num * num - 1]
      ans = max(ans, summ)

    return ans


# Link: https://leetcode.com/problems/maximum-element-sum-of-a-complete-subset-of-indices/description/
class Solution:
  def maximumSum(self, nums: List[int]) -> int:
    ans = 0
    oddPowerToSum = collections.Counter()

    def divideSquares(val: int) -> int:
      for num in range(2, val + 1):
        while val % (num * num) == 0:
          val //= (num * num)
      return val

    for i, num in enumerate(nums):
      oddPower = divideSquares(i + 1)
      oddPowerToSum[oddPower] += num
      ans = max(ans, oddPowerToSum[oddPower])

    return ans


# Link: https://leetcode.com/problems/split-the-array-to-make-coprime-products/description/
class Solution:
  def findValidSplit(self, nums: List[int]) -> int:
    leftPrimeFactors = collections.Counter()
    rightPrimeFactors = collections.Counter()

    def getPrimeFactors(num: int) -> List[int]:
      """Gets the prime factors under sqrt(10^6)."""
      primeFactors = []
      for divisor in range(2, min(1000, num) + 1):
        if num % divisor == 0:
          primeFactors.append(divisor)
          while num % divisor == 0:
            num //= divisor
      # Handle the case that `num` contains a prime factor > 1000.
      if num > 1:
        primeFactors.append(num)
      return primeFactors

    for num in nums:
      for primeFactor in getPrimeFactors(num):
        rightPrimeFactors[primeFactor] += 1

    for i in range(len(nums) - 1):
      for primeFactor in getPrimeFactors(nums[i]):
        rightPrimeFactors[primeFactor] -= 1
        if rightPrimeFactors[primeFactor] == 0:
          # rightPrimeFactors[primeFactor] == 0, so no need to track
          # leftPrimeFactors[primeFactor].
          del rightPrimeFactors[primeFactor]
          del leftPrimeFactors[primeFactor]
        else:
          # Otherwise, need to track leftPrimeFactors[primeFactor].
          leftPrimeFactors[primeFactor] += 1
      if not leftPrimeFactors:
        return i

    return -1


# Link: https://leetcode.com/problems/basic-calculator-iv/description/
class Poly:
  def __init__(self, term: str = None, coef: int = None):
    if term and coef:
      self.terms = collections.Counter({term: coef})
    else:
      self.terms = collections.Counter()

  def __add__(self, other):
    for term, coef in other.terms.items():
      self.terms[term] += coef
    return self

  def __sub__(self, other):
    for term, coef in other.terms.items():
      self.terms[term] -= coef
    return self

  def __mul__(self, other):
    res = Poly()
    for a, aCoef in self.terms.items():
      for b, bCoef in other.terms.items():
        res.terms[self._merge(a, b)] += aCoef * bCoef
    return res

  # Def __str__(self):
  #   res = []
  #   for term, coef in self.terms.items():
  #     res.append(term + ': ' + str(coef))
  #   return '{' + ', '.join(res) + '}'

  def toList(self) -> List[str]:
    for term in list(self.terms.keys()):
      if not self.terms[term]:
        del self.terms[term]

    def cmp(term: str) -> tuple:
      # the minimum degree is the last
      if term == '1':
        return (0,)
      var = term.split('*')
      # the maximum degree is the first
      # Break ties by their lexicographic orders.
      return (-len(var), term)

    def concat(term: str) -> str:
      if term == '1':
        return str(self.terms[term])
      return str(self.terms[term]) + '*' + term

    terms = list(self.terms.keys())
    terms.sort(key=cmp)
    return [concat(term) for term in terms]

  def _merge(self, a: str, b: str) -> str:
    if a == '1':
      return b
    if b == '1':
      return a
    res = []
    A = a.split('*')
    B = b.split('*')
    i = 0  # A's index
    j = 0  # B's index
    while i < len(A) and j < len(B):
      if A[i] < B[j]:
        res.append(A[i])
        i += 1
      else:
        res.append(B[j])
        j += 1
    return '*'.join(res + A[i:] + B[j:])


class Solution:
  def basicCalculatorIV(self, expression: str, evalvars: List[str], evalints: List[int]) -> List[str]:
    tokens = list(self._getTokens(expression))
    evalMap = {a: b for a, b in zip(evalvars, evalints)}

    for i, token in enumerate(tokens):
      if token in evalMap:
        tokens[i] = str(evalMap[token])

    postfix = self._infixToPostfix(tokens)
    return self._evaluate(postfix).toList()

  def _getTokens(self, s: str) -> Iterator[str]:
    i = 0
    for j, c in enumerate(s):
      if c == ' ':
        if i < j:
          yield s[i:j]
        i = j + 1
      elif c in '()+-*':
        if i < j:
          yield s[i:j]
        yield c
        i = j + 1
    if i < len(s):
      yield s[i:]

  def _infixToPostfix(self, tokens: List[str]) -> List[str]:
    postfix = []
    ops = []

    def precedes(prevOp: str, currOp: str) -> bool:
      if prevOp == '(':
        return False
      return prevOp == '*' or currOp in '+-'

    for token in tokens:
      if token == '(':
        ops.append(token)
      elif token == ')':
        while ops[-1] != '(':
          postfix.append(ops.pop())
        ops.pop()
      elif token in '+-*':  # isOperator(token)
        while ops and precedes(ops[-1], token):
          postfix.append(ops.pop())
        ops.append(token)
      else:  # isOperand(token)
        postfix.append(token)
    return postfix + ops[::-1]

  def _evaluate(self, postfix: List[str]) -> Poly:
    polys: List[Poly] = []
    for token in postfix:
      if token in '+-*':
        b = polys.pop()
        a = polys.pop()
        if token == '+':
          polys.append(a + b)
        elif token == '-':
          polys.append(a - b)
        else:  # token == '*'
          polys.append(a * b)
      elif token.lstrip('-').isnumeric():
        polys.append(Poly("1", int(token)))
      else:
        polys.append(Poly(token, 1))
    return polys[0]


# Link: https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/description/
# The read4 API is already defined for you.
# Def read4(buf4: List[str]) -> int:

class Solution:
  def read(self, buf: List[str], n: int) -> int:
    i = 0  # buf's index

    while i < n:
      if self.i4 == self.n4:  # All the characters in the buf4 are consumed.
        self.i4 = 0  # Reset the buf4's index.
        # Read <= 4 characters from the file to the buf4.
        self.n4 = read4(self.buf4)
        if self.n4 == 0:  # Reach the EOF.
          return i
      buf[i] = self.buf4[self.i4]
      i += 1
      self.i4 += 1

    return i

  buf4 = [' '] * 4
  i4 = 0  # buf4's index
  n4 = 0  # buf4's size


# Link: https://leetcode.com/problems/minimum-number-of-refueling-stops/description/
class Solution:
  def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
    # dp[i] := the farthest position we can reach w / i refuels
    dp = [startFuel] + [0] * len(stations)

    for i, station in enumerate(stations):
      for j in range(i + 1, 0, -1):
        if dp[j - 1] >= station[0]:
          dp[j] = max(dp[j], dp[j - 1] + station[1])

    for i, d in enumerate(dp):
      if d >= target:
        return i

    return -1


# Link: https://leetcode.com/problems/minimum-number-of-refueling-stops/description/
class Solution:
  def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
    ans = 0
    i = 0  # station's index
    curr = startFuel
    maxHeap = []

    while curr < target:
      # Add all the reachable stops to maxHeap
      while i < len(stations) and stations[i][0] <= curr:
        heapq.heappush(maxHeap, -stations[i][1])
        i += 1
      if not maxHeap:  # Can't be refueled.
        return -1
      curr -= heapq.heappop(maxHeap)  # Pop out the largest gas.
      ans += 1  # Then, refuel once.

    return ans


# Link: https://leetcode.com/problems/perfect-rectangle/description/
class Solution:
  def isRectangleCover(self, rectangles: List[List[int]]) -> bool:
    area = 0
    x1 = math.inf
    y1 = math.inf
    x2 = -math.inf
    y2 = -math.inf
    corners: Set[Tuple[int, int]] = set()

    for x, y, a, b in rectangles:
      area += (a - x) * (b - y)
      x1 = min(x1, x)
      y1 = min(y1, y)
      x2 = max(x2, a)
      y2 = max(y2, b)

      # the four points of the current rectangle
      for point in [(x, y), (x, b), (a, y), (a, b)]:
        if point in corners:
          corners.remove(point)
        else:
          corners.add(point)

    if len(corners) != 4:
      return False
    if (x1, y1) not in corners or \
        (x1, y2) not in corners or \
        (x2, y1) not in corners or \
            (x2, y2) not in corners:
      return False
    return area == (x2 - x1) * (y2 - y1)


# Link: https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/description/
class Solution:
  @functools.lru_cache(None)
  def rearrangeSticks(self, n: int, k: int) -> int:
    if n == k:
      return 1
    if k == 0:
      return 0
    return (self.rearrangeSticks(n - 1, k - 1) +
            self.rearrangeSticks(n - 1, k) * (n - 1)) % self.kMod

  kMod = 1_000_000_007


# Link: https://leetcode.com/problems/shortest-cycle-in-a-graph/description/
class Solution:
  def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
    kInf = 1001
    ans = kInf
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    def bfs(i: int) -> int:
      """Returns the length of the minimum cycle by starting BFS from node `i`.

      Returns `kInf` if there's no cycle.
      """
      dist = [kInf] * n
      q = collections.deque([i])
      dist[i] = 0
      while q:
        u = q.popleft()
        for v in graph[u]:
          if dist[v] == kInf:
            dist[v] = dist[u] + 1
            q.append(v)
          elif dist[v] + 1 != dist[u]:   # v is not a parent u.
            return dist[v] + dist[u] + 1
      return kInf

    ans = min(map(bfs, range(n)))
    return -1 if ans == kInf else ans


# Link: https://leetcode.com/problems/check-if-point-is-reachable/description/
class Solution:
  def isReachable(self, targetX: int, targetY: int) -> bool:
    return math.gcd(targetX, targetY).bit_count() == 1


# Link: https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/description/
class Solution:
  def minKBitFlips(self, nums: List[int], k: int) -> int:
    ans = 0
    flippedTime = 0

    for i, num in enumerate(nums):
      if i >= k and nums[i - k] == 2:
        flippedTime -= 1
      if flippedTime % 2 == num:
        if i + k > len(nums):
          return -1
        ans += 1
        flippedTime += 1
        nums[i] = 2

    return ans


# Link: https://leetcode.com/problems/split-array-with-same-average/description/
class Solution:
  def splitArraySameAverage(self, nums: List[int]) -> bool:
    n = len(nums)
    summ = sum(nums)
    if not any(i * summ % n == 0 for i in range(1, n // 2 + 1)):
      return False

    sums = [set() for _ in range(n // 2 + 1)]
    sums[0].add(0)

    for num in nums:
      for i in range(n // 2, 0, -1):
        for val in sums[i - 1]:
          sums[i].add(num + val)

    for i in range(1, n // 2 + 1):
      if i * summ % n == 0 and i * summ // n in sums[i]:
        return True

    return False


# Link: https://leetcode.com/problems/apply-operations-to-maximize-frequency-score/description/
class Solution:
  def maxFrequencyScore(self, nums: List[int], k: int) -> int:
    nums.sort()
    ans = 0
    cost = 0

    l = 0
    for r, num in enumerate(nums):
      cost += num - nums[(l + r) // 2]
      while cost > k:
        cost -= nums[(l + r + 1) // 2] - nums[l]
        l += 1
      ans = max(ans, r - l + 1)

    return ans


# Link: https://leetcode.com/problems/minimum-operations-to-make-numbers-non-positive/description/
class Solution:
  def minOperations(self, nums: list[int], x: int, y: int) -> int:
    def isPossible(m: int) -> bool:
      """
      Returns True if it's possible to make all `nums` <= 0 using m operations.
      """
      # If we want m operations, first decrease all the numbers by y * m. Then
      # we have m operations to select indices to decrease them by x - y.
      return sum(max(0, math.ceil((num - y * m) / (x - y)))
                 for num in nums) <= m

    return bisect.bisect_left(range(max(nums)), True,
                              key=isPossible)


# Link: https://leetcode.com/problems/count-integers-in-intervals/description/
from sortedcontainers import SortedDict


class CountIntervals:
  def __init__(self):
    self.intervals = SortedDict()
    self.cnt = 0

  def add(self, left: int, right: int) -> None:
    while self._isOverlapped(left, right):
      i = self.intervals.bisect_right(right) - 1
      l, r = self.intervals.popitem(i)
      left = min(left, l)
      right = max(right, r)
      self.cnt -= r - l + 1

    self.intervals[left] = right
    self.cnt += right - left + 1

  def count(self) -> int:
    return self.cnt

  def _isOverlapped(self, left: int, right: int) -> bool:
    i = self.intervals.bisect_right(right)
    return i > 0 and self.intervals.peekitem(i - 1)[1] >= left


# Link: https://leetcode.com/problems/range-module/description/
class RangeModule:
  def __init__(self):
    self.A = []

  def addRange(self, left: int, right: int) -> None:
    i = bisect_left(self.A, left)
    j = bisect_right(self.A, right)
    self.A[i:j] = [left] * (i % 2 == 0) + [right] * (j % 2 == 0)

  def queryRange(self, left: int, right: int) -> bool:
    i = bisect_right(self.A, left)
    j = bisect_left(self.A, right)
    return i == j and i % 2 == 1

  def removeRange(self, left: int, right: int) -> None:
    i = bisect_left(self.A, left)
    j = bisect_right(self.A, right)
    self.A[i:j] = [left] * (i % 2 == 1) + [right] * (j % 2 == 1)


# Link: https://leetcode.com/problems/integer-to-english-words/description/
class Solution:
  def numberToWords(self, num: int) -> str:
    if num == 0:
      return "Zero"

    belowTwenty = ["",        "One",       "Two",      "Three",
                   "Four",    "Five",      "Six",      "Seven",
                   "Eight",   "Nine",      "Ten",      "Eleven",
                   "Twelve",  "Thirteen",  "Fourteen", "Fifteen",
                   "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["",      "Ten",   "Twenty",  "Thirty", "Forty",
            "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def helper(num: int) -> str:
      if num < 20:
        s = belowTwenty[num]
      elif num < 100:
        s = tens[num // 10] + " " + belowTwenty[num % 10]
      elif num < 1000:
        s = helper(num // 100) + " Hundred " + helper(num % 100)
      elif num < 1000000:
        s = helper(num // 1000) + " Thousand " + helper(num % 1000)
      elif num < 1000000000:
        s = helper(num // 1000000) + " Million " + \
            helper(num % 1000000)
      else:
        s = helper(num // 1000000000) + " Billion " + \
            helper(num % 1000000000)

      return s.strip()

    return helper(num)


# Link: https://leetcode.com/problems/grid-illumination/description/
class Solution:
  def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
    ans = []
    rows = collections.Counter()
    cols = collections.Counter()
    diag1 = collections.Counter()
    diag2 = collections.Counter()
    lampsSet = set()

    for i, j in lamps:
      if (i, j) not in lampsSet:
        lampsSet.add((i, j))
        rows[i] += 1
        cols[j] += 1
        diag1[i + j] += 1
        diag2[i - j] += 1

    for i, j in queries:
      if rows[i] or cols[j] or diag1[i + j] or diag2[i - j]:
        ans.append(1)
        for y in range(max(0, i - 1), min(n, i + 2)):
          for x in range(max(0, j - 1), min(n, j + 2)):
            if (y, x) in lampsSet:
              lampsSet.remove((y, x))
              rows[y] -= 1
              cols[x] -= 1
              diag1[y + x] -= 1
              diag2[y - x] -= 1
      else:
        ans.append(0)

    return ans


# Link: https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/description/
from sortedcontainers import SortedList


class Solution:
  def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
    # Equivalently, the problem is to find nums[0] + the minimum sum of the top
    # k - 1 numbers in nums[i..i + dist], where i > 0 and i + dist < n.
    windowSum = sum(nums[i] for i in range(1, dist + 2))
    selected = SortedList(nums[i] for i in range(1, dist + 2))
    candidates = SortedList()

    def balance() -> int:
      """
      Returns the updated `windowSum` by balancing the multiset `selected` to
      keep the top k - 1 numbers.
      """
      nonlocal windowSum
      while len(selected) < k - 1:
        minCandidate = candidates[0]
        windowSum += minCandidate
        selected.add(minCandidate)
        candidates.remove(minCandidate)
      while len(selected) > k - 1:
        maxSelected = selected[-1]
        windowSum -= maxSelected
        selected.remove(maxSelected)
        candidates.add(maxSelected)
      return windowSum

    windowSum = balance()
    minWindowSum = windowSum

    for i in range(dist + 2, len(nums)):
      outOfScope = nums[i - dist - 1]
      if outOfScope in selected:
        windowSum -= outOfScope
        selected.remove(outOfScope)
      else:
        candidates.remove(outOfScope)
      if nums[i] < selected[-1]:  # nums[i] is a better number.
        windowSum += nums[i]
        selected.add(nums[i])
      else:
        candidates.add(nums[i])
      windowSum = balance()
      minWindowSum = min(minWindowSum, windowSum)

    return nums[0] + minWindowSum


# Link: https://leetcode.com/problems/last-substring-in-lexicographical-order/description/
class Solution:
  def lastSubstring(self, s: str) -> str:
    i = 0
    j = 1
    k = 0  # the number of the same letters of s[i..n) and s[j..n)

    while j + k < len(s):
      if s[i + k] == s[j + k]:
        k += 1
      elif s[i + k] > s[j + k]:
        # s[i..i + k] == s[j..j + k] and s[i + k] > s[j + k] means that we
        # should start from s[j + k] to find a possible larger substring.
        j += k + 1
        k = 0
      else:
        # s[i..i + k] == s[j..j + k] and s[i + k] < s[j + k] means that either
        # starting from s[i + k + 1] or s[j] has a larger substring
        i = max(i + k + 1, j)
        j = i + 1
        k = 0

    return s[i:]


# Link: https://leetcode.com/problems/permutation-sequence/description/
class Solution:
  def getPermutation(self, n: int, k: int) -> str:
    ans = ''
    nums = [i + 1 for i in range(n)]
    fact = [1] * (n + 1)  # fact[i] := i!

    for i in range(2, n + 1):
      fact[i] = fact[i - 1] * i

    k -= 1  # 0-indexed

    for i in reversed(range(n)):
      j = k // fact[i]
      k %= fact[i]
      ans += str(nums[j])
      nums.pop(j)

    return ans


# Link: https://leetcode.com/problems/minimum-cost-to-make-array-equal/description/
class Solution:
  def minCost(self, nums: List[int], cost: List[int]) -> int:
    ans = 0
    l = min(nums)
    r = max(nums)

    def getCost(target: int) -> int:
      return sum(abs(num - target) * c for num, c in zip(nums, cost))

    while l < r:
      m = (l + r) // 2
      cost1 = getCost(m)
      cost2 = getCost(m + 1)
      ans = min(cost1, cost2)
      if cost1 < cost2:
        r = m
      else:
        l = m + 1

    return ans


# Link: https://leetcode.com/problems/the-wording-game/description/
class Solution:
  def canAliceWin(self, a: List[str], b: List[str]) -> bool:
    # words[0][i] := the biggest word starting with ('a' + i) for Alice
    # words[1][i] := the biggest word starting with ('a' + i) for Bob
    words = [[''] * 26 for _ in range(2)]

    # For each letter, only the biggest word is useful.
    for word in a:
      words[0][ord(word[0]) - ord('a')] = word

    for word in b:
      words[1][ord(word[0]) - ord('a')] = word

    # Find Alice's smallest word.
    i = 0
    while not words[0][i]:
      i += 1

    # 0 := Alice, 1 := Bob
    # Start with Alice, so it's Bob's turn now.
    turn = 1

    # Iterate each letter until we find a winner.
    while True:
      # If the current player has a word that having the letter that is greater
      # than the opponent's word, choose it.
      if words[turn][i] and words[turn][i] > words[1 - turn][i]:
        # Choose the current words[turn][i].
        pass
      elif words[turn][i + 1]:
        # Choose the next words[turn][i + 1].
        i += 1
      else:
        # Game over. If it's Bob's turn, Alice wins, and vice versa.
        return turn == 1
      turn = 1 - turn


# Link: https://leetcode.com/problems/strange-printer-ii/description/
from enum import Enum


class State(Enum):
  kInit = 0
  kVisiting = 1
  kVisited = 2


class Solution:
  def isPrintable(self, targetGrid: List[List[int]]) -> bool:
    kMaxColor = 60
    m = len(targetGrid)
    n = len(targetGrid[0])

    # graph[u] := {v1, v2} means v1 and v2 cover u
    graph = [set() for _ in range(kMaxColor + 1)]

    for color in range(1, kMaxColor + 1):
      # Get the rectangle of the current color.
      minI = m
      minJ = n
      maxI = -1
      maxJ = -1
      for i in range(m):
        for j in range(n):
          if targetGrid[i][j] == color:
            minI = min(minI, i)
            minJ = min(minJ, j)
            maxI = max(maxI, i)
            maxJ = max(maxJ, j)

      # Add any color covering the current as the children.
      for i in range(minI, maxI + 1):
        for j in range(minJ, maxJ + 1):
          if targetGrid[i][j] != color:
            graph[color].add(targetGrid[i][j])

    states = [State.kInit] * (kMaxColor + 1)

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

    return not (any(hasCycle(i) for i in range(1, kMaxColor + 1)))


# Link: https://leetcode.com/problems/basic-calculator/description/
class Solution:
  def calculate(self, s: str) -> int:
    ans = 0
    num = 0
    sign = 1
    stack = [sign]  # stack[-1]: the current environment's sign

    for c in s:
      if c.isdigit():
        num = num * 10 + (ord(c) - ord('0'))
      elif c == '(':
        stack.append(sign)
      elif c == ')':
        stack.pop()
      elif c == '+' or c == '-':
        ans += sign * num
        sign = (1 if c == '+' else -1) * stack[-1]
        num = 0

    return ans + sign * num


# Link: https://leetcode.com/problems/count-houses-in-a-circular-street-ii/description/
# Definition for a street.
# class Street:
#   def closeDoor(self):
#     pass
#   def isDoorOpen(self):
#     pass
#   def moveRight(self):
#     pass
class Solution:
  def houseCount(self, street: Optional['Street'], k: int) -> int:
    ans = 0

    # Go to the first open door.
    while not street.isDoorOpen():
      street.moveRight()

    street.moveRight()

    for count in range(k):
      # Each time we encounter an open door, there's a possibility that it's the
      # first open door we intentionally left open.
      if street.isDoorOpen():
        ans = count + 1
        street.closeDoor()
      street.moveRight()

    return ans


# Link: https://leetcode.com/problems/kth-smallest-product-of-two-sorted-arrays/description/
class Solution:
  def kthSmallestProduct(self, nums1: List[int], nums2: List[int], k: int) -> int:
    A1 = [-num for num in nums1 if num < 0][::-1]  # Reverse to sort ascending
    A2 = [num for num in nums1 if num >= 0]
    B1 = [-num for num in nums2 if num < 0][::-1]  # Reverse to sort ascending
    B2 = [num for num in nums2 if num >= 0]

    negCount = len(A1) * len(B2) + len(A2) * len(B1)

    if k > negCount:  # Find (k - negCount)-th positive
      k -= negCount
      sign = 1
    else:
      k = negCount - k + 1  # Find (negCount - k + 1)-th abs(negative).
      sign = -1
      B1, B2 = B2, B1

    def numProductNoGreaterThan(A: List[int], B: List[int], m: int) -> int:
      ans = 0
      j = len(B) - 1
      for i in range(len(A)):
        # For each A[i], find the first index j s.t. A[i] * B[j] <= m
        # So numProductNoGreaterThan m for this row will be j + 1
        while j >= 0 and A[i] * B[j] > m:
          j -= 1
        ans += j + 1
      return ans

    l = 0
    r = 10**10

    while l < r:
      m = (l + r) // 2
      if numProductNoGreaterThan(A1, B1, m) + \
              numProductNoGreaterThan(A2, B2, m) >= k:
        r = m
      else:
        l = m + 1

    return sign * l


# Link: https://leetcode.com/problems/maximum-xor-of-two-non-overlapping-subtrees/description/
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
  def maxXor(self, n: int, edges: List[List[int]], values: List[int]) -> int:
    ans = 0
    tree = [[] for _ in range(n)]
    treeSums = [0] * n

    for u, v in edges:
      tree[u].append(v)
      tree[v].append(u)

    # Gets the tree sum rooted at node u.
    def getTreeSum(u: int, prev: int) -> int:
      treeSum = values[u]
      for v in tree[u]:
        if v == prev:
          continue
        treeSum += getTreeSum(v, u)
      treeSums[u] = treeSum
      return treeSum

    def dfs(u: int, prev: int, bitTrie: BitTrie) -> None:
      nonlocal ans
      for v in tree[u]:
        if v == prev:
          continue
        # Preorder to get the ans.
        ans = max(ans, bitTrie.getMaxXor(treeSums[v]))
        # Recursively call on the subtree rooted at node v.
        dfs(v, u, bitTrie)
        # Postorder to insert the tree sum rooted at node v.
        bitTrie.insert(treeSums[v])

    getTreeSum(0, -1)
    maxBit = int(math.log2(max(treeSums[1:])))
    # Similar to 421. Maximum XOR of Two Numbers in an Array
    dfs(0, -1, BitTrie(maxBit))
    return ans


# Link: https://leetcode.com/problems/find-shortest-path-with-k-hops/description/
class Solution:
  # Similar to 787. Cheapest Flights Within K Stops
  def shortestPathWithHops(self, n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
    graph = [[] for _ in range(n)]

    for u, v, w in edges:
      graph[u].append((v, w))
      graph[v].append((u, w))

    return self._dijkstra(graph, s, d, k)

  def _dijkstra(self, graph: List[List[Tuple[int, int]]], src: int, dst: int, k: int) -> int:
    dist = [[math.inf for _ in range(k + 1)] for _ in range(len(graph))]

    dist[src][k] = 0
    minHeap = [(dist[src][k], src, k)]  # (d, u, hops)

    while minHeap:
      d, u, hops = heapq.heappop(minHeap)
      if u == dst:
        return d
      for v, w in graph[u]:
        # Go from u -> v with w cost.
        if d + w < dist[v][hops]:
          dist[v][hops] = d + w
          heapq.heappush(minHeap, (dist[v][hops], v, hops))
        # Hop from u -> v with 0 cost.
        if hops > 0 and d < dist[v][hops - 1]:
          dist[v][hops - 1] = d
          heapq.heappush(minHeap, (dist[v][hops - 1], v, hops - 1))


# Link: https://leetcode.com/problems/count-the-number-of-incremovable-subarrays-ii/description/
class Solution:
  # Same as 2970. Count the Number of Incremovable Subarrays I
  def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n = len(nums)
    startIndex = self._getStartIndexOfSuffix(nums)
    # If the complete array is strictly increasing, the total number of ways we
    # can remove elements equals the total number of possible subarrays.
    if startIndex == 0:
      return n * (n + 1) // 2

    # The valid removals starting from nums[0] include nums[0..startIndex - 1],
    # nums[0..startIndex], ..., nums[0..n).
    ans = n - startIndex + 1

    # Enumerate each prefix subarray that is strictly increasing.
    j = startIndex
    for i in range(startIndex):
      if i > 0 and nums[i] <= nums[i - 1]:
        break
      # Since nums[0..i] is strictly increasing, move j to the place such that
      # nums[j] > nums[i]. The valid removals will then be nums[i + 1..j - 1],
      # nums[i + 1..j], ..., nums[i + 1..n).
      while j < n and nums[i] >= nums[j]:
        j += 1
      ans += n - j + 1

    return ans

  def _getStartIndexOfSuffix(self, nums: List[int]) -> int:
    for i in range(len(nums) - 2, -1, -1):
      if nums[i] >= nums[i + 1]:
        return i + 1
    return 0


# Link: https://leetcode.com/problems/count-the-number-of-incremovable-subarrays-ii/description/
class Solution:
  # Same as 2970. Count the Number of Incremovable Subarrays I
  def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n = len(nums)
    startIndex = self._getStartIndexOfSuffix(nums)
    # If the complete array is strictly increasing, the total number of ways we
    # can remove elements equals the total number of possible subarrays.
    if startIndex == 0:
      return n * (n + 1) // 2

    # The valid removals starting from nums[0] include nums[0..startIndex - 1],
    # nums[0..startIndex], ..., nums[0..n).
    ans = n - startIndex + 1

    # Enumerate each prefix subarray that is strictly increasing.
    for i in range(startIndex):
      if i > 0 and nums[i] <= nums[i - 1]:
        break
      # Since nums[0..i] is strictly increasing, find the first index j in
      # nums[startIndex..n) such that nums[j] > nums[i]. The valid removals
      # will then be nums[i + 1..j - 1], nums[i + 1..j], ..., nums[i + 1..n).
      ans += n - bisect.bisect_right(nums, nums[i], startIndex) + 1

    return ans

  def _getStartIndexOfSuffix(self, nums: List[int]) -> int:
    for i in range(len(nums) - 2, -1, -1):
      if nums[i] >= nums[i + 1]:
        return i + 1
    return 0


# Link: https://leetcode.com/problems/maximum-and-sum-of-array/description/
class Solution:
  def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
    n = 2 * numSlots
    nSelected = 1 << n
    # dp[i] := the maximum value, where i is the bitmask of the selected
    # numbers
    dp = [0] * nSelected

    nums += [0] * (n - len(nums))

    for mask in range(1, nSelected):
      selected = mask.bit_count()
      slot = (selected + 1) // 2  # (1, 2) -> 1, (3, 4) -> 2
      for i, num in enumerate(nums):
        if mask >> i & 1:  # Assign `nums[i]` to the `slot`-th slot.
          dp[mask] = max(dp[mask], dp[mask ^ 1 << i] + (slot & num))

    return dp[-1]


# Link: https://leetcode.com/problems/substring-with-largest-variance/description/
class Solution:
  def largestVariance(self, s: str) -> int:
    # a := the letter with the higher frequency
    # b := the letter with the lower frequency
    def kadane(a: str, b: str) -> int:
      ans = 0
      countA = 0
      countB = 0
      canExtendPrevB = False

      for c in s:
        if c != a and c != b:
          continue
        if c == a:
          countA += 1
        else:
          countB += 1
        if countB > 0:
          # An interval should contain at least one b.
          ans = max(ans, countA - countB)
        elif countB == 0 and canExtendPrevB:
          # edge case: consider the previous b.
          ans = max(ans, countA - 1)
        # Reset if the number of b > the number of a.
        if countB > countA:
          countA = 0
          countB = 0
          canExtendPrevB = True

      return ans

    return max(kadane(a, b)
               for a in string.ascii_lowercase
               for b in string.ascii_lowercase
               if a != b)


# Link: https://leetcode.com/problems/palindrome-partitioning-iv/description/
class Solution:
  def checkPartitioning(self, s: str) -> bool:
    n = len(s)
    # dp[i][j] := true if s[i..j] is a palindrome
    dp = [[False] * n for _ in range(n)]

    for i in range(n):
      dp[i][i] = True

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        if s[i] == s[j]:
          dp[i][j] = i + 1 > j - 1 or dp[i + 1][j - 1]

    for i in range(n):
      for j in range(i + 1, n):
        if dp[0][i] and dp[i + 1][j] and dp[j + 1][n - 1]:
          return True

    return False


# Link: https://leetcode.com/problems/palindrome-partitioning-iv/description/
class Solution:
  def checkPartitioning(self, s: str) -> bool:
    @functools.lru_cache(None)
    def isPalindrome(i: int, j: int) -> bool:
      """Returns True if s[i..j] is a palindrome."""
      if i > j:
        return True
      if s[i] == s[j]:
        return isPalindrome(i + 1, j - 1)
      return False

    n = len(s)
    return any(isPalindrome(0, i) and
               isPalindrome(i + 1, j) and
               isPalindrome(j + 1, n - 1)
               for i in range(n)
               for j in range(i + 1, n - 1))


# Link: https://leetcode.com/problems/coin-path/description/
class Solution:
  def cheapestJump(self, coins: List[int], maxJump: int) -> List[int]:
    if coins[-1] == -1:
      return []

    n = len(coins)
    # dp[i] := the minimum cost to jump from i to n - 1
    dp = [math.inf] * n
    next = [-1] * n

    def cheapestJump(i: int) -> int:
      if i == len(coins) - 1:
        dp[i] = coins[i]
        return dp[i]
      if dp[i] < math.inf:
        return dp[i]
      if coins[i] == -1:
        return math.inf

      for j in range(i + 1, min(i + maxJump + 1, n)):
        res = cheapestJump(j)
        if res == math.inf:
          continue
        cost = coins[i] + res
        if cost < dp[i]:
          dp[i] = cost
          next[i] = j

      return dp[i]

    cheapestJump(0)
    if dp[0] == math.inf:
      return []
    return self._constructPath(next, 0)

  def _constructPath(self, next: List[int], i: int) -> List[int]:
    ans = []
    while i != -1:
      ans.append(i + 1)  # 1-indexed
      i = next[i]
    return ans


# Link: https://leetcode.com/problems/coin-path/description/
class Solution:
  def cheapestJump(self, coins: List[int], maxJump: int) -> List[int]:
    if coins[-1] == -1:
      return []

    n = len(coins)
    # dp[i] := the minimum cost to jump from i to n - 1
    dp = [math.inf] * n
    next = [-1] * n

    dp[-1] = coins[-1]

    for i in reversed(range(n - 1)):
      if coins[i] == -1:
        continue
      for j in range(i + 1, min(i + maxJump + 1, n)):
        if dp[j] == math.inf:
          continue
        cost = coins[i] + dp[j]
        if cost < dp[i]:
          dp[i] = cost
          next[i] = j

    if dp[0] == math.inf:
      return []
    return self._constructPath(next, 0)

  def _constructPath(self, next: List[int], i: int) -> List[int]:
    ans = []
    while i != -1:
      ans.append(i + 1)  # 1-indexed
      i = next[i]
    return ans


# Link: https://leetcode.com/problems/jump-game-iv/description/
class Solution:
  def minJumps(self, arr: List[int]) -> int:
    n = len(arr)
    # {num: indices}
    graph = collections.defaultdict(list)
    step = 0
    q = collections.deque([0])
    seen = {0}

    for i, a in enumerate(arr):
      graph[a].append(i)

    while q:
      for _ in range(len(q)):
        i = q.popleft()
        if i == n - 1:
          return step
        seen.add(i)
        u = arr[i]
        if i + 1 < n:
          graph[u].append(i + 1)
        if i - 1 >= 0:
          graph[u].append(i - 1)
        for v in graph[u]:
          if v in seen:
            continue
          q.append(v)
        graph[u].clear()
      step += 1


# Link: https://leetcode.com/problems/minimum-time-for-k-virus-variants-to-spread/description/
class Solution:
  def minDayskVariants(self, points: List[List[int]], k: int) -> int:
    kMax = 100
    ans = math.inf

    for a in range(1, kMax + 1):
      for b in range(1, kMax + 1):
        # Stores the k minimum distances of points that can reach (a, b).
        maxHeap = []
        for x, y in points:
          heapq.heappush(maxHeap, -abs(x - a) + -abs(y - b))
          if len(maxHeap) > k:
            heapq.heappop(maxHeap)
        ans = min(ans, -maxHeap[0])

    return ans


# Link: https://leetcode.com/problems/check-for-contradictions-in-equations/description/
class Solution:
  def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
    # Convert `string` to `int` for a better perfermance.
    strToInt = {}

    for u, v in equations:
      strToInt.setdefault(u, len(strToInt))
      strToInt.setdefault(v, len(strToInt))

    graph = [[] for _ in range(len(strToInt))]
    seen = [0.0] * len(graph)

    for i, ((A, B), value) in enumerate(zip(equations, values)):
      u = strToInt[A]
      v = strToInt[B]
      graph[u].append((v, value))
      graph[v].append((u, 1 / value))

    def dfs(u: int, val: float) -> bool:
      if seen[u]:
        return abs(val / seen[u] - 1) > 1e-5

      seen[u] = val
      return any(dfs(v, val / w) for v, w in graph[u])

    for i in range(len(graph)):
      if not seen[i] and dfs(i, 1.0):
        return True

    return False


# Link: https://leetcode.com/problems/24-game/description/
class Solution:
  def judgePoint24(self, nums: List[int]) -> bool:
    def generate(a: float, b: float) -> List[float]:
      return [a * b,
              math.inf if b == 0 else a / b,
              math.inf if a == 0 else b / a,
              a + b, a - b, b - a]

    def dfs(nums: List[float]) -> bool:
      if len(nums) == 1:
        return abs(nums[0] - 24.0) < 0.001

      for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
          for num in generate(nums[i], nums[j]):
            nextRound = [num]
            for k in range(len(nums)):
              if k == i or k == j:
                continue
              nextRound.append(nums[k])
            if dfs(nextRound):
              return True

      return False

    return dfs(nums)


# Link: https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/description/
class Solution:
  def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
    @lru_cache(None)
    def height(root: Optional[TreeNode]) -> int:
      if not root:
        return 0
      return 1 + max(height(root.left), height(root.right))

    # valToMaxHeight[val] := the maximum height without the node with `val`
    valToMaxHeight = {}

    # maxHeight := the maximum height without the current node `root`
    def dfs(root: Optional[TreeNode], depth: int, maxHeight: int) -> None:
      if not root:
        return
      valToMaxHeight[root.val] = maxHeight
      dfs(root.left, depth + 1, max(maxHeight, depth + height(root.right)))
      dfs(root.right, depth + 1, max(maxHeight, depth + height(root.left)))

    dfs(root, 0, 0)
    return [valToMaxHeight[query] for query in queries]


# Link: https://leetcode.com/problems/minimum-space-wasted-from-packaging/description/
class Solution:
  def minWastedSpace(self, packages: List[int], boxes: List[List[int]]) -> int:
    ans = math.inf

    packages.sort()

    for box in boxes:
      box.sort()
      if box[-1] < packages[-1]:
        continue
      accu = 0
      i = 0
      for b in box:
        j = bisect.bisect(packages, b, i)
        accu += b * (j - i)
        i = j
      ans = min(ans, accu)

    return -1 if ans == math.inf else (ans - sum(packages)) % int(1e9 + 7)


# Link: https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/description/
class T:
  def __init__(self, isBST: bool = False,
               max: Optional[int] = None,
               min: Optional[int] = None,
               sum: Optional[int] = None):
    self.isBST = isBST
    self.max = max
    self.min = min
    self.sum = sum


class Solution:
  def maxSumBST(self, root: Optional[TreeNode]) -> int:
    self.ans = 0

    def traverse(root: Optional[TreeNode]) -> T:
      if not root:
        return T(True, -math.inf, math.inf, 0)

      left: T = traverse(root.left)
      right: T = traverse(root.right)

      if not left.isBST or not right.isBST:
        return T()
      if root.val <= left.max or root.val >= right.min:
        return T()

      # The `root` is a valid BST.
      summ = root.val + left.sum + right.sum
      self.ans = max(self.ans, summ)
      return T(True, max(root.val, right.max), min(root.val, left.min), summ)

    traverse(root)
    return self.ans


# Link: https://leetcode.com/problems/distinct-echo-substrings/description/
class Solution:
  def distinctEchoSubstrings(self, text: str) -> int:
    seen = set()

    for k in range(1, len(text) // 2 + 1):  # the target length
      same = 0
      l = 0
      for r in range(k, len(text)):
        if text[l] == text[r]:
          same += 1
        else:
          same = 0
        if same == k:
          seen.add(text[l - k + 1:l + 1])
          # Move the window thus leaving a letter behind, so we need to
          # decrease the counter.
          same -= 1
        l += 1

    return len(seen)


# Link: https://leetcode.com/problems/erect-the-fence/description/
class Solution:
  def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
    hull = []

    trees.sort(key=lambda x: (x[0], x[1]))

    def cross(p: List[int], q: List[int], r: List[int]) -> int:
      return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # Build the lower hull: left-to-right scan.
    for tree in trees:
      while len(hull) > 1 and cross(hull[-1], hull[-2], tree) > 0:
        hull.pop()
      hull.append(tuple(tree))
    hull.pop()

    # Build the upper hull: right-to-left scan.
    for tree in reversed(trees):
      while len(hull) > 1 and cross(hull[-1], hull[-2], tree) > 0:
        hull.pop()
      hull.append(tuple(tree))

    # Remove the redundant elements from the stack.
    return list(set(hull))


# Link: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
class Solution:
  def maxProfit(self, k: int, prices: List[int]) -> int:
    if k >= len(prices) // 2:
      sell = 0
      hold = -math.inf

      for price in prices:
        sell = max(sell, hold + price)
        hold = max(hold, sell - price)

      return sell

    sell = [0] * (k + 1)
    hold = [-math.inf] * (k + 1)

    for price in prices:
      for i in range(k, 0, -1):
        sell[i] = max(sell[i], hold[i] + price)
        hold[i] = max(hold[i], sell[i - 1] - price)

    return sell[k]


# Link: https://leetcode.com/problems/count-paths-that-can-form-a-palindrome-in-a-tree/description/
class Solution:
  def countPalindromePaths(self, parent: List[int], s: str) -> int:
    # A valid (u, v) has at most 1 letter with odd frequency on its path. The
    # frequency of a letter on the u-v path is equal to the sum of its
    # frequencies on the root-u and root-v paths substract twice of its
    # frequency on the root-LCA(u, v) path. Considering only the parity
    # (even/odd), the part involving root-LCA(u, v) can be ignored, making it
    # possible to calculate both parts easily using a simple DFS.
    tree = collections.defaultdict(list)
    maskToCount = collections.Counter({0: 1})

    for i in range(1, len(parent)):
      tree[parent[i]].append(i)

    # mask := 26 bits that represent the parity of each character in the alphabet
    # on the path from node 0 to node u
    def dfs(u: int, mask: int) -> int:
      res = 0
      if u > 0:
        mask ^= 1 << (ord(s[u]) - ord('a'))
        # Consider any u-v path with 1 bit set.
        for i in range(26):
          res += maskToCount[mask ^ (1 << i)]
        # Consider u-v path with 0 bit set.
        res += maskToCount[mask ^ 0]
        maskToCount[mask] += 1
      for v in tree[u]:
        res += dfs(v, mask)
      return res

    return dfs(0, 0)


# Link: https://leetcode.com/problems/find-a-good-subset-of-the-matrix/description/
class Solution:
  def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
    kMaxBit = 30
    maskToIndex = {}

    def getMask(row: List[int]) -> int:
      mask = 0
      for i, num in enumerate(row):
        if num == 1:
          mask |= 1 << i
      return mask

    for i, row in enumerate(grid):
      mask = getMask(row)
      if mask == 0:
        return [i]
      for prevMask in range(1, kMaxBit):
        if (mask & prevMask) == 0 and prevMask in maskToIndex:
          return [maskToIndex[prevMask], i]
      maskToIndex[mask] = i

    return []


# Link: https://leetcode.com/problems/count-palindromic-subsequences/description/
class Solution:
  def countPalindromes(self, s: str) -> int:
    kMod = 1_000_000_007
    ans = 0

    for a in range(10):
      for b in range(10):
        pattern = f'{a}{b}.{b}{a}'
        # dp[i] := the number of subsequences of pattern[i..n) in s, where
        # pattern[2] can be any character
        dp = [0] * 5 + [1]
        for c in s:
          for i, p in enumerate(pattern):
            if p == '.' or p == c:
              dp[i] += dp[i + 1]
        ans += dp[0]
        ans %= kMod

    return ans


# Link: https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/description/
class Solution:
  def waysToPartition(self, nums: List[int], k: int) -> int:
    n = len(nums)
    summ = sum(nums)
    prefix = 0
    # Count of sum(A[0..k)) - sum(A[k..n)) for k in [0..i)
    l = collections.Counter()
    # Count of sum(A[0..k)) - sum(A[k..n)) for k in [i..n)
    r = collections.Counter()

    for pivot in range(1, n):
      prefix += nums[pivot - 1]
      suffix = summ - prefix
      r[prefix - suffix] += 1

    ans = r[0]
    prefix = 0

    for num in nums:
      ans = max(ans, l[k - num] + r[num - k])
      prefix += num
      suffix = summ - prefix
      diff = prefix - suffix
      r[diff] -= 1
      l[diff] += 1

    return ans


# Link: https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/description/
class Solution:
  def countOrders(self, n: int) -> int:
    kMod = 1_000_000_007
    ans = 1

    for i in range(1, n + 1):
      ans = ans * i * (i * 2 - 1) % kMod

    return ans


# Link: https://leetcode.com/problems/find-in-mountain-array/description/
# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class MountainArray:
#   def get(self, index: int) -> int:
#   def length(self) -> int:

class Solution:
  def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
    n = mountain_arr.length()
    peakIndex = self.peakIndexInMountainArray(mountain_arr, 0, n - 1)

    leftIndex = self.searchLeft(mountain_arr, target, 0, peakIndex)
    if mountain_arr.get(leftIndex) == target:
      return leftIndex

    rightIndex = self.searchRight(mountain_arr, target, peakIndex + 1, n - 1)
    if mountain_arr.get(rightIndex) == target:
      return rightIndex

    return -1

  # 852. Peak Index in a Mountain Array
  def peakIndexInMountainArray(self, A: 'MountainArray', l: int, r: int) -> int:
    while l < r:
      m = (l + r) // 2
      if A.get(m) < A.get(m + 1):
        l = m + 1
      else:
        r = m
    return l

  def searchLeft(self, A: 'MountainArray', target: int, l: int, r: int) -> int:
    while l < r:
      m = (l + r) // 2
      if A.get(m) < target:
        l = m + 1
      else:
        r = m
    return l

  def searchRight(self, A: 'MountainArray', target: int, l: int, r: int) -> int:
    while l < r:
      m = (l + r) // 2
      if A.get(m) > target:
        l = m + 1
      else:
        r = m
    return l


# Link: https://leetcode.com/problems/count-anagrams/description/
class Solution:
  def countAnagrams(self, s: str) -> int:
    ans = 1

    for word in s.split():
      ans = ans * math.factorial(len(word))
      count = collections.Counter(word)
      for freq in count.values():
        ans //= math.factorial(freq)

    return ans % 1_000_000_007


# Link: https://leetcode.com/problems/closest-binary-search-tree-value-ii/description/
class Solution:
  def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
    dq = collections.deque()

    def inorder(root: Optional[TreeNode]) -> None:
      if not root:
        return

      inorder(root.left)
      dq.append(root.val)
      inorder(root.right)

    inorder(root)

    while len(dq) > k:
      if abs(dq[0] - target) > abs(dq[-1] - target):
        dq.popleft()
      else:
        dq.pop()

    return list(dq)


# Link: https://leetcode.com/problems/painting-a-grid-with-three-different-colors/description/
class Solution:
  def colorTheGrid(self, m: int, n: int) -> int:
    def getColor(mask: int, r: int) -> int:
      return mask >> r * 2 & 3

    def setColor(mask: int, r: int, color: int) -> int:
      return mask | color << r * 2

    kMod = 1_000_000_007

    @functools.lru_cache(None)
    def dp(r: int, c: int, prevColMask: int, currColMask: int) -> int:
      if c == n:
        return 1
      if r == m:
        return dp(0, c + 1, currColMask, 0)

      ans = 0

      # 1 := red, 2 := green, 3 := blue
      for color in range(1, 4):
        if getColor(prevColMask, r) == color:
          continue
        if r > 0 and getColor(currColMask, r - 1) == color:
          continue
        ans += dp(r + 1, c, prevColMask, setColor(currColMask, r, color))
        ans %= kMod

      return ans

    return dp(0, 0, 0, 0)


# Link: https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
class Solution:
  def maxPathSum(self, root: Optional[TreeNode]) -> int:
    ans = -math.inf

    def maxPathSumDownFrom(root: Optional[TreeNode]) -> int:
      """
      Returns the maximum path sum starting from the current root, where
      root.val is always included.
      """
      nonlocal ans
      if not root:
        return 0

      l = max(0, maxPathSumDownFrom(root.left))
      r = max(0, maxPathSumDownFrom(root.right))
      ans = max(ans, root.val + l + r)
      return root.val + max(l, r)

    maxPathSumDownFrom(root)
    return ans


# Link: https://leetcode.com/problems/naming-a-company/description/
class Solution:
  def distinctNames(self, ideas: List[str]) -> int:
    ans = 0
    # suffixes[i] := the set of strings omitting the first letter, where the
    # first letter is ('a' + i)
    suffixes = [set() for _ in range(26)]

    for idea in ideas:
      suffixes[ord(idea[0]) - ord('a')].add(idea[1:])

    for i in range(25):
      for j in range(i + 1, 26):
        count = len(suffixes[i] & suffixes[j])
        ans += 2 * (len(suffixes[i]) - count) * (len(suffixes[j]) - count)

    return ans


# Link: https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/description/
class Solution:
  def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
    @functools.lru_cache(None)
    def dp(mask: int) -> int:
      i = bin(mask).count('1')
      if i == len(nums1):
        return 0
      return min((nums1[i] ^ nums2[j]) + dp(mask | 1 << j)
                 for j in range(len(nums2)) if not mask >> j & 1)
    return dp(0)


# Link: https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/
class Solution:
  def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
    ans = 0
    j = -1
    prevMinKIndex = -1
    prevMaxKIndex = -1

    for i, num in enumerate(nums):
      if num < minK or num > maxK:
        j = i
      if num == minK:
        prevMinKIndex = i
      if num == maxK:
        prevMaxKIndex = i
      # Any index k in [j + 1, min(prevMinKIndex, prevMaxKIndex)] can be the
      # start of the subarray s.t. nums[k..i] satisfies the conditions.
      ans += max(0, min(prevMinKIndex, prevMaxKIndex) - j)

    return ans


# Link: https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and/description/
class Solution:
  def getXORSum(self, arr1: List[int], arr2: List[int]) -> int:
    return functools.reduce(operator.xor, arr1) & functools.reduce(operator.xor, arr2)


# Link: https://leetcode.com/problems/employee-free-time/description/
class Solution:
  def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
    ans = []
    intervals = []

    for s in schedule:
      intervals.extend(s)

    intervals.sort(key=lambda x: x.start)

    prevEnd = intervals[0].end

    for interval in intervals:
      if interval.start > prevEnd:
        ans.append(Interval(prevEnd, interval.start))
      prevEnd = max(prevEnd, interval.end)

    return ans


# Link: https://leetcode.com/problems/count-different-palindromic-subsequences/description/
class Solution:
  def countPalindromicSubsequences(self, s: str) -> int:
    kMod = 1_000_000_007
    n = len(s)
    # dp[i][j] := the number of different non-empty palindromic subsequences in
    # s[i..j]
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
      dp[i][i] = 1

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        if s[i] == s[j]:
          lo = i + 1
          hi = j - 1
          while lo <= hi and s[lo] != s[i]:
            lo += 1
          while lo <= hi and s[hi] != s[i]:
            hi -= 1
          if lo > hi:
            dp[i][j] = dp[i + 1][j - 1] * 2 + 2
          elif lo == hi:
            dp[i][j] = dp[i + 1][j - 1] * 2 + 1
          else:
            dp[i][j] = dp[i + 1][j - 1] * 2 - dp[lo + 1][hi - 1]
        else:
          dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1]
        dp[i][j] = (dp[i][j] + kMod) % kMod

    return dp[0][n - 1]


# Link: https://leetcode.com/problems/basic-calculator-iii/description/
class Solution:
  def calculate(self, s: str) -> int:
    nums = []
    ops = []

    def calc():
      b = nums.pop()
      a = nums.pop()
      op = ops.pop()
      if op == '+':
        nums.append(a + b)
      elif op == '-':
        nums.append(a - b)
      elif op == '*':
        nums.append(a * b)
      else:  # op == '/'
        nums.append(int(a / b))

    def precedes(prev: str, curr: str) -> bool:
      """
      Returns True if the previous character is a operator and the priority of
      the previous operator >= the priority of the current character (operator).
      """
      if prev == '(':
        return False
      return prev in '*/' or curr in '+-'

    i = 0
    hasPrevNum = False

    while i < len(s):
      c = s[i]
      if c.isdigit():
        num = ord(c) - ord('0')
        while i + 1 < len(s) and s[i + 1].isdigit():
          num = num * 10 + (ord(s[i + 1]) - ord('0'))
          i += 1
        nums.append(num)
        hasPrevNum = True
      elif c == '(':
        ops.append('(')
        hasPrevNum = False
      elif c == ')':
        while ops[-1] != '(':
          calc()
        ops.pop()  # Pop '('
      elif c in '+-*/':
        if not hasPrevNum:  # Handle input like "-1-(-1)"
          num.append(0)
        while ops and precedes(ops[-1], c):
          calc()
        ops.append(c)
      i += 1

    while ops:
      calc()

    return nums.pop()


# Link: https://leetcode.com/problems/count-vowels-permutation/description/
class Solution:
  def countVowelPermutation(self, n: int) -> int:
    kMod = 1_000_000_007
    dp = {'a': 1, 'e': 1, 'i': 1, 'o': 1, 'u': 1}

    for _ in range(n - 1):
      newDp = {'a': dp['e'] + dp['i'] + dp['u'],
               'e': dp['a'] + dp['i'],
               'i': dp['e'] + dp['o'],
               'o': dp['i'],
               'u': dp['i'] + dp['o']}
      dp = newDp

    return sum(dp.values()) % kMod


# Link: https://leetcode.com/problems/dice-roll-simulation/description/
class Solution:
  def dieSimulator(self, n: int, rollMax: List[int]) -> int:
    kMaxRolls = 15
    kMod = 1_000_000_007

    dp = [[[0] * (kMaxRolls + 1) for j in range(6)] for i in range(n + 1)]

    for num in range(6):
      dp[1][num][1] = 1

    for i in range(2, n + 1):
      for currNum in range(6):
        for prevNum in range(6):
          for k in range(1, 15 + 1):
            if prevNum != currNum:
              dp[i][currNum][1] = (
                  dp[i][currNum][1] + dp[i - 1][prevNum][k]) % kMod
            elif k < rollMax[currNum]:
              dp[i][currNum][k + 1] = dp[i - 1][currNum][k]

    ans = 0

    for num in range(6):
      for k in range(1, 15 + 1):
        ans += dp[n][num][k]

    return ans % kMod


# Link: https://leetcode.com/problems/maximum-frequency-stack/description/
class FreqStack:
  def __init__(self):
    self.maxFreq = 0
    self.count = collections.Counter()
    self.countToStack = collections.defaultdict(list)

  def push(self, val: int) -> None:
    self.count[val] += 1
    self.countToStack[self.count[val]].append(val)
    self.maxFreq = max(self.maxFreq, self.count[val])

  def pop(self) -> int:
    val = self.countToStack[self.maxFreq].pop()
    self.count[val] -= 1
    if not self.countToStack[self.maxFreq]:
      self.maxFreq -= 1
    return val


# Link: https://leetcode.com/problems/find-pattern-in-infinite-stream-ii/description/
# Definition for an infinite stream.
# class InfiniteStream:
#   def next(self) -> int:
#     pass

class Solution:
  # Same as 3023. Find Pattern in Infinite Stream I
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


# Link: https://leetcode.com/problems/distinct-subsequences-ii/description/
class Solution:
  def distinctSubseqII(self, s: str) -> int:
    kMod = 1_000_000_007
    # endsIn[i] := the number of subsequence that end in ('a' + i)
    endsIn = [0] * 26

    for c in s:
      endsIn[ord(c) - ord('a')] = (sum(endsIn) + 1) % kMod

    return sum(endsIn) % kMod


# Link: https://leetcode.com/problems/minimum-falling-path-sum-ii/description/
class Solution:
  def minFallingPathSum(self, grid: List[List[int]]) -> int:
    n = len(grid)

    for i in range(1, n):
      (firstMinNum, firstMinIndex), (secondMinNum, _) = sorted(
          {(a, i) for i, a in enumerate(grid[i - 1])})[:2]
      for j in range(n):
        if j == firstMinIndex:
          grid[i][j] += secondMinNum
        else:
          grid[i][j] += firstMinNum

    return min(grid[-1])


# Link: https://leetcode.com/problems/minimum-time-to-revert-word-to-initial-state-ii/description/
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


# Link: https://leetcode.com/problems/parsing-a-boolean-expression/description/
class Solution:
  def parseBoolExpr(self, expression: str) -> bool:
    def dfs(s: int, e: int) -> List[str]:
      if s == e:
        return True if expression[s] == 't' else False

      exps = []
      layer = 0

      for i in range(s, e + 1):
        c = expression[i]
        if layer == 0 and c in '!&|':
          op = c
        elif c == '(':
          layer += 1
          if layer == 1:
            left = i + 1
        elif c == ')':
          layer -= 1
          if layer == 0:
            exps.append(dfs(left, i - 1))
        elif c == ',' and layer == 1:
          exps.append(dfs(left, i - 1))
          left = i + 1

      if op == '|':
        return functools.reduce(lambda x, y: x | y, exps)
      if op == '&':
        return functools.reduce(lambda x, y: x & y, exps)
      if op == '!':
        return not exps[0]

    return dfs(0, len(expression) - 1)


# Link: https://leetcode.com/problems/find-median-from-data-stream/description/
class MedianFinder:
  def __init__(self):
    self.maxHeap = []
    self.minHeap = []

  def addNum(self, num: int) -> None:
    if not self.maxHeap or num <= -self.maxHeap[0]:
      heapq.heappush(self.maxHeap, -num)
    else:
      heapq.heappush(self.minHeap, num)

    # Balance the two heaps s.t.
    # |maxHeap| >= |minHeap| and |maxHeap| - |minHeap| <= 1.
    if len(self.maxHeap) < len(self.minHeap):
      heapq.heappush(self.maxHeap, -heapq.heappop(self.minHeap))
    elif len(self.maxHeap) - len(self.minHeap) > 1:
      heapq.heappush(self.minHeap, -heapq.heappop(self.maxHeap))

  def findMedian(self) -> float:
    if len(self.maxHeap) == len(self.minHeap):
      return (-self.maxHeap[0] + self.minHeap[0]) / 2.0
    return -self.maxHeap[0]


# Link: https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/description/
class Solution:
  # Similar to 907. Sum of Subarray Minimums
  def validSubarraySize(self, nums: List[int], threshold: int) -> int:
    n = len(nums)
    ans = 0
    # prev[i] := the index k s.t. nums[k] is the previous minimum in nums[0..n)
    prev = [-1] * n
    # next[i] := the index k s.t. nums[k] is the next minimum in nums[i + 1..n)
    next = [n] * n
    stack = []

    for i, a in enumerate(nums):
      while stack and nums[stack[-1]] > a:
        index = stack.pop()
        next[index] = i
      if stack:
        prev[i] = stack[-1]
      stack.append(i)

    for i, (num, prevIndex, nextIndex) in enumerate(zip(nums, prev, next)):
      k = (i - prevIndex) + (nextIndex - i) - 1
      if num > threshold / k:
        return k

    return -1


# Link: https://leetcode.com/problems/numbers-at-most-n-given-digit-set/description/
class Solution:
  def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
    ans = 0
    num = str(n)

    for i in range(1, len(num)):
      ans += pow(len(digits), i)

    for i, c in enumerate(num):
      dHasSameNum = False
      for digit in digits:
        if digit[0] < c:
          ans += pow(len(digits), len(num) - i - 1)
        elif digit[0] == c:
          dHasSameNum = True
      if not dHasSameNum:
        return ans

    return ans + 1


# Link: https://leetcode.com/problems/distinct-subsequences/description/
class Solution:
  def numDistinct(self, s: str, t: str) -> int:
    m = len(s)
    n = len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
      dp[i][0] = 1

    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if s[i - 1] == t[j - 1]:
          dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
        else:
          dp[i][j] = dp[i - 1][j]

    return dp[m][n]


# Link: https://leetcode.com/problems/distinct-subsequences/description/
class Solution:
  def numDistinct(self, s: str, t: str) -> int:
    m = len(s)
    n = len(t)
    dp = [1] + [0] * n

    for i in range(1, m + 1):
      for j in range(n, 1 - 1, -1):
        if s[i - 1] == t[j - 1]:
          dp[j] += dp[j - 1]

    return dp[n]


# Link: https://leetcode.com/problems/reverse-subarray-to-maximize-array-value/description/
class Solution:
  def maxValueAfterReverse(self, nums: List[int]) -> int:
    mini = math.inf
    maxi = -math.inf

    for a, b in zip(nums, nums[1:]):
      mini = min(mini, max(a, b))
      maxi = max(maxi, min(a, b))
    diff = max(0, (maxi - mini) * 2)

    for a, b in zip(nums, nums[1:]):
      headDiff = -abs(a - b) + abs(nums[0] - b)
      tailDiff = -abs(a - b) + abs(nums[-1] - a)
      diff = max(diff, headDiff, tailDiff)

    return sum(abs(a - b) for a, b in zip(nums, nums[1:])) + diff


# Link: https://leetcode.com/problems/stamping-the-sequence/description/
class Solution:
  def movesToStamp(self, stamp: str, target: str) -> List[int]:
    def stampify(s: int) -> int:
      """
      Stamps target[i..i + |stamp|) and returns the number of newly stamped
      characters.
      e.g. stampify("abc", "ababc", 2) returns 3 because target becomes "ab***".
      """
      stampified = len(stamp)

      for i, st in enumerate(stamp):
        if target[s + i] == '*':  # It's already been stamped.
          stampified -= 1
        elif target[s + i] != st:  # We can't stamp on the index i.
          return 0

      for i in range(s, s + len(stamp)):
        target[i] = '*'

      return stampified

    ans = []
    target = list(target)
    # stamped[i] := True if we already stamped target by stamping on index i
    stamped = [False] * len(target)
    stampedCount = 0  # Our goal is to make stampedCount = |target|.

    while stampedCount < len(target):
      isStamped = False
      # Try to stamp target[i..i + |stamp|) for each index.
      for i in range(len(target) - len(stamp) + 1):
        if stamped[i]:
          continue
        stampified = stampify(i)
        if stampified == 0:
          continue
        stampedCount += stampified
        isStamped = True
        stamped[i] = True
        ans.append(i)
      # After trying to stamp on each index, we can't find a valid stamp.
      if not isStamped:
        return []

    return ans[::-1]


# Link: https://leetcode.com/problems/number-of-possible-sets-of-closing-branches/description/
class Solution:
  def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
    return sum(self._floydWarshall(n, maxDistance, roads, mask) <= maxDistance
               for mask in range(1 << n))

  def _floydWarshall(self, n: int, maxDistanceThreshold: int,
                     roads: List[List[int]], mask: int) -> List[List[int]]:
    """
    Returns the maximum distance between any two branches, where the mask
    represents the selected branches.
    """
    maxDistance = 0
    dist = [[maxDistanceThreshold + 1] * n for _ in range(n)]

    for i in range(n):
      if mask >> i & 1:
        dist[i][i] = 0

    for u, v, w in roads:
      if mask >> u & 1 and mask >> v & 1:
        dist[u][v] = min(dist[u][v], w)
        dist[v][u] = min(dist[v][u], w)

    for k in range(n):
      if mask >> k & 1:
        for i in range(n):
          if mask >> i & 1:
            for j in range(n):
              if mask >> j & 1:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    for i in range(n):
      if mask >> i & 1:
        for j in range(i + 1, n):
          if mask >> j & 1:
            maxDistance = max(maxDistance, dist[i][j])

    return maxDistance


# Link: https://leetcode.com/problems/super-washing-machines/description/
class Solution:
  def findMinMoves(self, machines: List[int]) -> int:
    dresses = sum(machines)

    if dresses % len(machines) != 0:
      return -1

    ans = 0
    average = dresses // len(machines)
    inout = 0

    for dress in machines:
      inout += dress - average
      ans = max(ans, abs(inout), dress - average)

    return ans


# Link: https://leetcode.com/problems/divide-chocolate/description/
class Solution:
  def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
    l = len(sweetness) // (k + 1)
    r = sum(sweetness) // (k + 1)

    def canEat(m: int) -> bool:
      """
      Returns True if can eat m sweetness (the minimum sweetness of each piece).
      """
      pieces = 0
      summ = 0  # the running sum
      for s in sweetness:
        summ += s
        if summ >= m:
          pieces += 1
          summ = 0
      return pieces > k

    while l < r:
      m = (l + r) // 2
      if canEat(m):
        l = m + 1
      else:
        r = m

    return l if canEat(l) else l - 1


# Link: https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix/description/
class Solution:
  def minFlips(self, mat: List[List[int]]) -> int:
    m = len(mat)
    n = len(mat[0])
    hash = self._getHash(mat, m, n)
    if hash == 0:
      return 0

    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    step = 0
    q = collections.deque([hash])
    seen = {hash}

    while q:
      step += 1
      for _ in range(len(q)):
        curr = q.popleft()
        for i in range(m):
          for j in range(n):
            next = curr ^ 1 << (i * n + j)
            # Flie the four neighbors.
            for dx, dy in dirs:
              x = i + dx
              y = j + dy
              if x < 0 or x == m or y < 0 or y == n:
                continue
              next ^= 1 << (x * n + y)
            if next == 0:
              return step
            if next in seen:
              continue
            q.append(next)
            seen.add(next)

    return -1

  def _getHash(self, mat: List[List[int]], m: int, n: int) -> int:
    hash = 0
    for i in range(m):
      for j in range(n):
        if mat[i][j]:
          hash |= 1 << (i * n + j)
    return hash


# Link: https://leetcode.com/problems/sum-of-k-mirror-numbers/description/
class Solution:
  def kMirror(self, k: int, n: int) -> int:
    ans = 0
    A = ['0']

    def nextKMirror(A: List[chr]) -> List[chr]:
      for i in range(len(A) // 2, len(A)):
        nextNum = int(A[i]) + 1
        if nextNum < k:
          A[i] = str(nextNum)
          A[~i] = str(nextNum)
          for j in range(len(A) // 2, i):
            A[j] = '0'
            A[~j] = '0'
          return A
      return ['1'] + ['0'] * (len(A) - 1) + ['1']

    for _ in range(n):
      while True:
        A = nextKMirror(A)
        num = int(''.join(A), k)
        if str(num)[::-1] == str(num):
          break
      ans += num

    return ans


# Link: https://leetcode.com/problems/encode-n-ary-tree-to-binary-tree/description/
class Codec:
  # Encodes an n-ary tree to a binary tree.
  def encode(self, root: 'Node') -> Optional[TreeNode]:
    if not root:
      return None

    rootTreeNode = TreeNode(root.val)
    q = collections.deque([(root, rootTreeNode)])

    while q:
      parentNode, parentTreeNode = q.popleft()
      prevTreeNode = None
      headTreeNode = None
      for child in parentNode.children:
        currTreeNode = TreeNode(child.val)
        if prevTreeNode:
          prevTreeNode.right = currTreeNode
        else:
          headTreeNode = currTreeNode
        prevTreeNode = currTreeNode
        q.append((child, currTreeNode))
      parentTreeNode.left = headTreeNode

    return rootTreeNode

  # Decodes your binary tree to an n-ary tree.
  def decode(self, root: Optional[TreeNode]) -> 'Node':
    if not root:
      return None

    rootNode = Node(root.val, [])
    q = collections.deque([(rootNode, root)])

    while q:
      parentNode, parentTreeNode = q.popleft()
      sibling = parentTreeNode.left
      while sibling:
        currNode = Node(sibling.val, [])
        parentNode.children.append(currNode)
        q.append((currNode, sibling))
        sibling = sibling.right

    return rootNode


# Link: https://leetcode.com/problems/encode-n-ary-tree-to-binary-tree/description/
class Codec:
  # Encodes an n-ary tree to a binary tree.
  def encode(self, root: 'Node') -> Optional[TreeNode]:
    if not root:
      return None

    rootTreeNode = TreeNode(root.val)
    if root.children:
      rootTreeNode.left = self.encode(root.children[0])

    # The parent for the rest of the children
    currTreeNode = rootTreeNode.left

    # Encode the rest of the children
    for i in range(1, len(root.children)):
      currTreeNode.right = self.encode(root.children[i])
      currTreeNode = currTreeNode.right

    return rootTreeNode

  # Decodes your binary tree to an n-ary tree.
  def decode(self, root: Optional[TreeNode]) -> 'Node':
    if not root:
      return None

    rootNode = Node(root.val, [])
    currTreeNode = root.left

    while currTreeNode:
      rootNode.children.append(self.decode(currTreeNode))
      currTreeNode = currTreeNode.right

    return rootNode


# Link: https://leetcode.com/problems/maximum-score-words-formed-by-letters/description/
class Solution:
  def maxScoreWords(self, words: List[str], letters: List[chr], score: List[int]) -> int:
    count = collections.Counter(letters)

    def useWord(i: int) -> int:
      isValid = True
      earned = 0
      for c in words[i]:
        count[c] -= 1
        if count[c] < 0:
          isValid = False
        earned += score[ord(c) - ord('a')]
      return earned if isValid else -1

    def unuseWord(i: int) -> None:
      for c in words[i]:
        count[c] += 1

    def dfs(s: int) -> int:
      """Returns the maximum score you can get from words[s..n)."""
      ans = 0
      for i in range(s, len(words)):
        earned = useWord(i)
        if earned > 0:
          ans = max(ans, earned + dfs(i + 1))
        unuseWord(i)
      return ans

    return dfs(0)


# Link: https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/description/
class Solution:
  def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
    n = len(floor)
    # dp[i][j] := the minimum number of visible white tiles of floor[i..n)
    # after covering at most j carpets
    dp = [[0] * (numCarpets + 1) for _ in range(n + 1)]

    for i in reversed(range(n)):
      dp[i][0] = int(floor[i]) + dp[i + 1][0]

    for i in reversed(range(n)):
      for j in range(1, numCarpets + 1):
        cover = dp[i + carpetLen][j - 1] if i + carpetLen < n else 0
        skip = int(floor[i]) + dp[i + 1][j]
        dp[i][j] = min(cover, skip)

    return dp[0][numCarpets]


# Link: https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/description/
class Solution:
  def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
    kMax = 1000

    @functools.lru_cache(None)
    def dp(i: int, j: int) -> int:
      """
      Returns the minimum number of visible white tiles of floor[i..n) after
      covering at most j carpets.
      """
      if j < 0:
        return kMax
      if i >= len(floor):
        return 0
      return min(dp(i + carpetLen, j - 1),
                 dp(i + 1, j) + int(floor[i]))

    return dp(0, numCarpets)


# Link: https://leetcode.com/problems/numbers-with-repeated-digits/description/
class Solution:
  def numDupDigitsAtMostN(self, n: int) -> int:
    return n - self._countSpecialNumbers(n)

  # Same as 2376. Count Special Integers
  def _countSpecialNumbers(self, n: int) -> int:
    s = str(n)

    @functools.lru_cache(None)
    def dp(i: int, used: int, isTight: bool) -> int:
      """
      Returns the number of special integers, considering the i-th digit, where
      `used` is the bitmask of the used digits, and `isTight` indicates if the
      current digit is tightly bound.
      """
      if i == len(s):
        return 1

      res = 0

      maxDigit = int(s[i]) if isTight else 9
      for d in range(maxDigit + 1):
        # `d` is used.
        if used >> d & 1:
          continue
        # Use `d` now.
        nextIsTight = isTight and (d == maxDigit)
        if used == 0 and d == 0:  # Don't count leading 0s as used.
          res += dp(i + 1, used, nextIsTight)
        else:
          res += dp(i + 1, used | 1 << d, nextIsTight)

      return res

    return dp(0, 0, True) - 1  # - 0


# Link: https://leetcode.com/problems/minimum-money-required-before-transactions/description/
class Solution:
  def minimumMoney(self, transactions: List[List[int]]) -> int:
    ans = 0
    losses = 0

    # Before picking the final transaction, perform any transaction that raises
    # the required money.
    for cost, cashback in transactions:
      losses += max(0, cost - cashback)

    # Now, pick a transaction to be the final one.
    for cost, cashback in transactions:
      if cost > cashback:
        # The losses except this transaction: losses - (cost - cashback), so
        # add the cost of this transaction = losses - (cost - cashback) + cost.
        ans = max(ans, losses + cashback)
      else:
        # The losses except this transaction: losses, so add the cost of this
        # transaction = losses + cost.
        ans = max(ans, losses + cost)

    return ans


# Link: https://leetcode.com/problems/put-marbles-in-bags/description/
class Solution:
  def putMarbles(self, weights: List[int], k: int) -> int:
    # To distribute marbles into k bags, there will be k - 1 cuts. If there's a
    # cut after weights[i], then weights[i] and weights[i + 1] will be added to
    # the cost. Also, no matter how we cut, weights[0] and weights[n - 1] will
    # be counted. So, the goal is to find the max//min k - 1 weights[i] +
    # weights[i + 1].

    # weights[i] + weights[i + 1]
    A = [a + b for a, b in itertools.pairwise(weights)]
    return sum(heapq.nlargest(k - 1, A)) - sum(heapq.nsmallest(k - 1, A))


# Link: https://leetcode.com/problems/maximum-points-after-collecting-coins-from-all-nodes/description/
class Solution:
  def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
    kMaxCoin = 10000
    kMaxHalved = int(kMaxCoin).bit_length()
    n = len(coins)
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    @functools.lru_cache(None)
    def dfs(u: int, prev: int, halved: int) -> int:
      # All the children will be 0, so no need to explore.
      if halved > kMaxHalved:
        return 0

      val = coins[u] // (1 << halved)
      takeAll = val - k
      takeHalf = math.floor(val / 2)

      for v in graph[u]:
        if v == prev:
          continue
        takeAll += dfs(v, u, halved)
        takeHalf += dfs(v, u, halved + 1)

      return max(takeAll, takeHalf)

    return dfs(0, -1, 0)


# Link: https://leetcode.com/problems/earliest-second-to-mark-indices-ii/description/
class Solution:
  def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
    # {the second: the index of nums can be zeroed at the current second}
    secondToIndex = self._getSecondToIndex(nums, changeIndices)
    numsSum = sum(nums)

    def canMark(maxSecond: int) -> bool:
      """
      Returns True if all indices of `nums` can be marked within `maxSecond`.
      """
      # Use a min-heap to greedily pop out the minimum number, which yields the
      # least saving.
      minHeap = []
      marks = 0

      for second in range(maxSecond - 1, -1, -1):
        if second in secondToIndex:
          # The number mapped by the index is a candidate to be zeroed out.
          index = secondToIndex[second]
          heapq.heappush(minHeap, nums[index])
          if marks == 0:
            # Running out of marks, so need to pop out the minimum number.
            # So, the current second will be used to mark an index.
            heapq.heappop(minHeap)
            marks += 1
          else:
            # There're enough marks.
            # So, the current second will be used to zero out a number.
            marks -= 1
        else:
          # There's no candidate to be zeroed out.
          # So, the current second will be used to mark an index.
          marks += 1

      decrementAndMarkCost = (numsSum - sum(minHeap)) + \
          (len(nums) - len(minHeap))
      zeroAndMarkCost = len(minHeap) + len(minHeap)
      return decrementAndMarkCost + zeroAndMarkCost <= maxSecond

    l = bisect.bisect_left(range(1, len(changeIndices) + 1), True,
                           key=lambda m: canMark(m)) + 1
    return l if l <= len(changeIndices) else -1

  def _getSecondToIndex(self, nums: List[int], changeIndices: List[int]) -> Dict[int, int]:
    # {the `index` of nums: the earliest second to zero out nums[index]}
    indexToFirstSecond = {}
    for zeroIndexedSecond, oneIndexedIndex in enumerate(changeIndices):
      index = oneIndexedIndex - 1  # Convert to 0-indexed.
      if nums[index] > 0 and index not in indexToFirstSecond:
        indexToFirstSecond[index] = zeroIndexedSecond
    return {second: index for index, second in indexToFirstSecond.items()}


# Link: https://leetcode.com/problems/triples-with-bitwise-and-equal-to-zero/description/
class Solution:
  def countTriplets(self, nums: List[int]) -> int:
    kMax = 1 << 16
    ans = 0
    count = [0] * kMax  # {nums[i] & nums[j]: times}

    for a in nums:
      for b in nums:
        count[a & b] += 1

    for num in nums:
      for i in range(kMax):
        if (num & i) == 0:
          ans += count[i]

    return ans


# Link: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/description/
class Solution:
  def maxProduct(self, s: str) -> int:
    n = len(s)

    def manacher(s: str) -> List[int]:
      maxExtends = [0] * n
      l2r = [1] * n
      center = 0

      for i in range(n):
        r = center + maxExtends[center] - 1
        mirrorIndex = center - (i - center)
        extend = 1 if i > r else min(maxExtends[mirrorIndex], r - i + 1)
        while i - extend >= 0 and i + extend < n and s[i - extend] == s[i + extend]:
          l2r[i + extend] = 2 * extend + 1
          extend += 1
        maxExtends[i] = extend
        if i + maxExtends[i] >= r:
          center = i

      for i in range(1, n):
        l2r[i] = max(l2r[i], l2r[i - 1])

      return l2r

    # l[i] := the maximum length of palindromes in s[0..i)
    l = manacher(s)
    # r[i] := the maximum length of palindromes in s[i..n)
    r = manacher(s[::-1])[::-1]
    return max(l[i] * r[i + 1] for i in range(n - 1))


# Link: https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-substrings/description/
class Solution:
  def maxProduct(self, s: str) -> int:
    kBase = 26
    kMod = 1_000_000_007
    n = len(s)
    ans = 1
    pow = [1] + [0] * n  # pow[i] := kBase^i
    hashFromL = [0] * (n + 1)  # hashFromL[i] = the hash of s[0..i)
    hashFromR = [0] * (n + 1)  # hashFromR[i] = the hash of s[i..n)
    l = [0] * n  # l[i] := the maximum length of palindromes in s[0..i)
    r = [0] * n  # r[i] := the maximum length of palindromes in s[i..n)

    for i in range(1, n + 1):
      pow[i] = pow[i - 1] * kBase % kMod

    def val(c: str) -> int:
      return ord(c) - ord('a')

    for i in range(1, n + 1):
      hashFromL[i] = (hashFromL[i - 1] * kBase + val(s[i - 1])) % kMod

    for i in reversed(range(n)):
      hashFromR[i] = (hashFromR[i + 1] * kBase + val(s[i])) % kMod

    # Returns the hash of s[l..r) from the left.
    def leftHash(l: int, r: int) -> int:
      hash = (hashFromL[r] - hashFromL[l] * pow[r - l]) % kMod
      return hash + kMod if hash < 0 else hash

    # Returns the hash of s[l..r) from the right.
    def rightHash(l: int, r: int) -> int:
      hash = (hashFromR[l] - hashFromR[r] * pow[r - l]) % kMod
      return hash + kMod if hash < 0 else hash

    # Returns true if s[l..r) is a palindrome.
    def isPalindrome(l: int, r: int) -> bool:
      return leftHash(l, r) == rightHash(l, r)

    maxi = 1  # the maximum length of palindromes so far
    for i in range(n):
      if i - maxi - 1 >= 0 and isPalindrome(i - maxi - 1, i + 1):
        maxi += 2
      l[i] = maxi

    # Fill in r.
    maxi = 1
    for i in reversed(range(n)):
      if i + maxi + 2 <= n and isPalindrome(i, i + maxi + 2):
        maxi += 2
      r[i] = maxi

    for i in range(n - 1):
      ans = max(ans, l[i] * r[i + 1])

    return ans


# Link: https://leetcode.com/problems/valid-number/description/
class Solution:
  def isNumber(self, s: str) -> bool:
    s = s.strip()
    if not s:
      return False

    seenNum = False
    seenDot = False
    seenE = False

    for i, c in enumerate(s):
      if c == '.':
        if seenDot or seenE:
          return False
        seenDot = True
      elif c == 'e' or c == 'E':
        if seenE or not seenNum:
          return False
        seenE = True
        seenNum = False
      elif c in '+-':
        if i > 0 and s[i - 1] not in 'eE':
          return False
        seenNum = False
      else:
        if not c.isdigit():
          return False
        seenNum = True

    return seenNum


# Link: https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/description/
class Solution:
  def maxValue(self, events: List[List[int]], k: int) -> int:
    events.sort()

    @functools.lru_cache(None)
    def dp(i: int, k: int) -> int:
      """
      Returns the maximum sum of values that you can receive by attending
      events[i..n), where k is the maximum number of attendance.
      """
      if k == 0 or i == len(events):
        return 0

      # Binary search `events` to find the first index j
      # s.t. events[j][0] > events[i][1].
      j = bisect.bisect(events, [events[i][1], math.inf, math.inf], i + 1)
      return max(events[i][2] + dp(j, k - 1), dp(i + 1, k))

    return dp(0, k)


# Link: https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/
class Solution:
  def numWays(self, words: List[str], target: str) -> int:
    kMod = 1_000_000_007
    wordLength = len(words[0])
    # counts[j] := the count map of words[i][j], where 0 <= i < |words|
    counts = [collections.Counter() for _ in range(wordLength)]

    for i in range(wordLength):
      for word in words:
        counts[i][word[i]] += 1

    @functools.lru_cache(None)
    def dp(i: int, j: int):
      """Returns the number of ways to form target[i..n) using word[j..n)."""
      if i == len(target):
        return 1
      if j == wordLength:
        return 0
      return (dp(i + 1, j + 1) * counts[j][target[i]] + dp(i, j + 1)) % kMod

    return dp(0, 0)


# Link: https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/
class Solution:
  def numWays(self, words: List[str], target: str) -> int:
    kMod = 1_000_000_007
    # dp[i] := the number of ways to form the first i characters of `target`
    dp = [0] * (len(target) + 1)
    dp[0] = 1

    for j in range(len(words[0])):
      count = collections.Counter(word[j] for word in words)
      for i in range(len(target), 0, -1):
        dp[i] += dp[i - 1] * count[target[i - 1]]
        dp[i] %= kMod

    return dp[len(target)]


# Link: https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/
class Solution:
  def numWays(self, words: List[str], target: str) -> int:
    kMod = 1_000_000_007
    wordLength = len(words[0])
    # dp[i][j] := the number of ways to form the first i characters of the
    # `target` using the j first characters in each word
    dp = [[0] * (wordLength + 1) for _ in range(len(target) + 1)]
    # counts[j] := the count map of words[i][j], where 0 <= i < |words|
    counts = [collections.Counter() for _ in range(wordLength)]

    for i in range(wordLength):
      for word in words:
        counts[i][word[i]] += 1

    dp[0][0] = 1

    for i in range(len(target) + 1):
      for j in range(wordLength):
        if i < len(target):
          # Pick the character target[i] from word[j].
          dp[i + 1][j + 1] = dp[i][j] * counts[j][target[i]]
          dp[i + 1][j + 1] %= kMod
        # Skip the word[j].
        dp[i][j + 1] += dp[i][j]
        dp[i][j + 1] %= kMod

    return dp[len(target)][wordLength]


# Link: https://leetcode.com/problems/design-excel-sum-formula/description/
class Cell:
  def __init__(self, val: int, posCount: Optional[Dict[Tuple[int, int], int]]):
    self.val = val
    self.posCount = posCount  # {pos, count}


class Excel:
  def __init__(self, height: int, width: str):
    self.sheet = [[Cell(0, None) for i in range(height)]
                  for _ in range(ord(width) - ord('A') + 1)]

  def set(self, row: int, column: str, val: int) -> None:
    self.sheet[row - 1][ord(column) - ord('A')] = Cell(val, None)

  def get(self, row: int, column: str) -> int:
    cell = self.sheet[row - 1][ord(column) - ord('A')]
    if cell.posCount:
      return sum(self.get(*pos) * freq for pos, freq in cell.posCount.items())
    return cell.val

  def sum(self, row: int, column: str, numbers: List[str]) -> int:
    self.sheet[row - 1][ord(column) - ord('A')].posCount = self._parse(numbers)
    return self.get(row, column)

  def _parse(self, numbers: List[str]):
    count = collections.Counter()
    for n in numbers:
      s, e = n.split(':')[0], n.split(':')[1] if ':' in n else n
      for i in range(int(s[1:]), int(e[1:]) + 1):
        for j in range(ord(s[0]) - ord('A'), ord(e[0]) - ord('A') + 1):
          count[(i, chr(j + ord('A')))] += 1
    return count


# Link: https://leetcode.com/problems/number-of-digit-one/description/
class Solution:
  def countDigitOne(self, n: int) -> int:
    ans = 0

    pow10 = 1
    while pow10 <= n:
      divisor = pow10 * 10
      quotient = n // divisor
      remainder = n % divisor
      if quotient > 0:
        ans += quotient * pow10
      if remainder >= pow10:
        ans += min(remainder - pow10 + 1, pow10)
      pow10 *= 10

    return ans


# Link: https://leetcode.com/problems/delete-duplicate-folders-in-system/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = collections.defaultdict(TrieNode)
    self.deleted = False


class Solution:
  def deleteDuplicateFolder(self, paths: List[List[str]]) -> List[List[str]]:
    ans = []
    root = TrieNode()
    subtreeToNodes: Dict[str, List[TrieNode]] = collections.defaultdict(list)

    # Construct the Trie
    for path in sorted(paths):
      node = root
      for s in path:
        node = node.children[s]

    # For each subtree, fill in the {subtree encoding: [root]} hash table
    def buildSubtreeToRoots(node: TrieNode) -> str:
      subtree = '(' + ''.join(s + buildSubtreeToRoots(node.children[s])
                              for s in node.children) + ')'
      if subtree != '()':
        subtreeToNodes[subtree].append(node)
      return subtree

    buildSubtreeToRoots(root)

    # Mark nodes that should be deleted
    for nodes in subtreeToNodes.values():
      if len(nodes) > 1:
        for node in nodes:
          node.deleted = True

    # Construct the answer array for nodes that haven't been deleted
    def constructPath(node: TrieNode, path: List[str]) -> None:
      for s, child in node.children.items():
        if not child.deleted:
          constructPath(child, path + [s])
      if path:
        ans.append(path)

    constructPath(root, [])
    return ans


# Link: https://leetcode.com/problems/poor-pigs/description/
class Solution:
  def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
    base = minutesToTest // minutesToDie + 1
    ans = 0
    x = 1
    while x < buckets:
      ans += 1
      x *= base
    return ans


# Link: https://leetcode.com/problems/confusing-number-ii/description/
class Solution:
  def confusingNumberII(self, n: int) -> int:
    digitToRotated = [(0, 0), (1, 1), (6, 9), (8, 8), (9, 6)]

    def dfs(num: int, rotatedNum: int, unit: int) -> int:
      ans = 0 if num == rotatedNum else 1
      # Add one more digit
      for digit, rotated in digitToRotated:
        if digit == 0 and num == 0:
          continue
        nextNum = num * 10 + digit
        if nextNum > n:
          break
        ans += dfs(nextNum, rotated * unit + rotatedNum, unit * 10)
      return ans

    return dfs(0, 0, 1)


# Link: https://leetcode.com/problems/sum-of-subsequence-widths/description/
class Solution:
  def sumSubseqWidths(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    ans = 0
    exp = 1

    nums.sort()

    for i in range(n):
      ans += (nums[i] - nums[n - 1 - i]) * exp
      ans %= kMod
      exp = exp * 2 % kMod

    return ans


# Link: https://leetcode.com/problems/palindrome-rearrangement-queries/description/
class Solution:
  def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
    n = len(s)
    # mirroredDiffs[i] := the number of different letters between the first i
    # letters of s[0..n / 2) and the first i letters of s[n / 2..n)[::-1]
    mirroredDiffs = self._getMirroredDiffs(s)
    # counts[i] := the count of s[0..i)
    counts = self._getCounts(s)
    ans = []

    def subtractArrays(a: List[int], b: List[int]):
      return [x - y for x, y in zip(a, b)]

    for a, b, c, d in queries:
      # Use left-closed, right-open intervals to facilitate the calculation.
      #   ...... [a, b) ...|... [rb, ra) ......
      #   .... [rd, rc) .....|..... [c, d) ....
      b += 1
      d += 1
      ra = n - a  # the reflected index of a in s[n / 2..n)
      rb = n - b  # the reflected index of b in s[n / 2..n)
      rc = n - c  # the reflected index of c in s[n / 2..n)
      rd = n - d  # the reflected index of d in s[n / 2..n)
      # No difference is allowed outside the query ranges.
      if (min(a, rd) > 0 and mirroredDiffs[min(a, rd)] > 0) or \
         (n // 2 > max(b, rc) and mirroredDiffs[n // 2] - mirroredDiffs[max(b, rc)] > 0) or \
         (rd > b and mirroredDiffs[rd] - mirroredDiffs[b] > 0) or \
         (a > rc and mirroredDiffs[a] - mirroredDiffs[rc] > 0):
        ans.append(False)
      else:
        # The `count` map of the intersection of [a, b) and [rd, rc) in
        # s[0..n / 2) must equate to the `count` map of the intersection of
        # [c, d) and [rb, ra) in s[n / 2..n).
        leftRangeCount = subtractArrays(counts[b], counts[a])
        rightRangeCount = subtractArrays(counts[d], counts[c])
        if a > rd:
          rightRangeCount = subtractArrays(
              rightRangeCount, subtractArrays(counts[min(a, rc)], counts[rd]))
        if rc > b:
          rightRangeCount = subtractArrays(
              rightRangeCount, subtractArrays(counts[rc], counts[max(b, rd)]))
        if c > rb:
          leftRangeCount = subtractArrays(
              leftRangeCount, subtractArrays(counts[min(c, ra)], counts[rb]))
        if ra > d:
          leftRangeCount = subtractArrays(
              leftRangeCount, subtractArrays(counts[ra], counts[max(d, rb)]))
        ans.append(min(leftRangeCount) >= 0
                   and min(rightRangeCount) >= 0
                   and leftRangeCount == rightRangeCount)

    return ans

  def _getMirroredDiffs(self, s: str) -> List[int]:
    diffs = [0]
    for i, j in zip(range(len(s)), reversed(range(len(s)))):
      if i >= j:
        break
      diffs.append(diffs[-1] + (s[i] != s[j]))
    return diffs

  def _getCounts(self, s: str) -> List[List[int]]:
    count = [0] * 26
    counts = [count.copy()]
    for c in s:
      count[ord(c) - ord('a')] += 1
      counts.append(count.copy())
    return counts


# Link: https://leetcode.com/problems/number-of-flowers-in-full-bloom/description/
class Solution:
  def fullBloomFlowers(self, flowers: List[List[int]], persons: List[int]) -> List[int]:
    starts = sorted(s for s, _ in flowers)
    ends = sorted(e for _, e in flowers)
    return [bisect.bisect_right(starts, person) -
            bisect.bisect_left(ends, person)
            for person in persons]


# Link: https://leetcode.com/problems/get-the-maximum-score/description/
class Solution:
  def maxSum(self, nums1: List[int], nums2: List[int]) -> int:
    # Keep the running the sum of `nums1` and `nums2` before the next rendezvous.
    # Since `nums1` and `nums2` are increasing, move forward on the smaller one
    # to ensure we don't miss any rendezvous. When meet rendezvous, choose the
    # better path.
    ans = 0
    sum1 = 0  # sum(nums1) in (the prevoious rendezvous, the next rendezvous)
    sum2 = 0  # sum(nums2) in (the prevoious rendezvous, the next rendezvous)
    i = 0  # nums1's index
    j = 0  # nums2's index

    while i < len(nums1) and j < len(nums2):
      if nums1[i] < nums2[j]:
        sum1 += nums1[i]
        i += 1
      elif nums1[i] > nums2[j]:
        sum2 += nums2[j]
        j += 1
      else:  # An rendezvous happens.
        ans += max(sum1, sum2) + nums1[i]
        sum1 = 0
        sum2 = 0
        i += 1
        j += 1

    while i < len(nums1):
      sum1 += nums1[i]
      i += 1

    while j < len(nums2):
      sum2 += nums2[j]
      j += 1

    return (ans + max(sum1, sum2)) % (10**9 + 7)


# Link: https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/description/
class Solution:
  def minimumObstacles(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    minHeap = [(grid[0][0], 0, 0)]  # (d, i, j)
    dist = [[math.inf] * n for _ in range(m)]
    dist[0][0] = grid[0][0]

    while minHeap:
      d, i, j = heapq.heappop(minHeap)
      if i == m - 1 and j == n - 1:
        return d
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        newDist = d + grid[i][j]
        if newDist < dist[x][y]:
          dist[x][y] = newDist
          heapq.heappush(minHeap, (newDist, x, y))

    return dist[m - 1][n - 1]


# Link: https://leetcode.com/problems/minimize-or-of-remaining-elements-using-operations/description/
class Solution:
  def minOrAfterOperations(self, nums: List[int], k: int) -> int:
    kMaxBit = 30
    ans = 0
    prefixMask = 0  # Grows like: 10000 -> 11000 -> ... -> 11111

    for i in range(kMaxBit, -1, -1):
      # Add the i-th bit to `prefixMask` and attempt to "turn off" the
      # currently added bit within k operations. If it's impossible, then we
      # add the i-th bit to the answer.
      prefixMask |= 1 << i
      if self._getMergeOps(nums, prefixMask, ans) > k:
        ans |= 1 << i

    return ans

  def _getMergeOps(self, nums: List[int], prefixMask: int, target: int) -> int:
    """
    Returns the number of merge operations to turn `prefixMask` to the target
    by ANDing `nums`.
    """
    mergeOps = 0
    ands = prefixMask
    for num in nums:
      ands &= num
      if (ands | target) == target:
        ands = prefixMask
      else:
        mergeOps += 1  # Keep merging the next num
    return mergeOps


# Link: https://leetcode.com/problems/minimum-operations-to-remove-adjacent-ones-in-matrix/description/
class Solution:
  def minimumOperations(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    seen = [[0] * n for _ in range(m)]
    match = [[-1] * n for _ in range(m)]

    def dfs(i: int, j: int, sessionId: int) -> int:
      for dx, dy in dirs:
        x = i + dx
        y = j + dy
        if x < 0 or x == m or y < 0 or y == n:
          continue
        if grid[x][y] == 0 or seen[x][y] == sessionId:
          continue
        seen[x][y] = sessionId
        if match[x][y] == -1 or dfs(*divmod(match[x][y], n), sessionId):
          match[x][y] = i * n + j
          match[i][j] = x * n + y
          return 1
      return 0

    ans = 0

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1 and match[i][j] == -1:
          sessionId = i * n + j
          seen[i][j] = sessionId
          ans += dfs(i, j, sessionId)

    return ans


# Link: https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/description/
class Solution:
  def minPushBox(self, grid: List[List[chr]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 'B':
          box = (i, j)
        elif grid[i][j] == 'S':
          player = (i, j)
        elif grid[i][j] == 'T':
          target = (i, j)

    def isInvalid(playerX: int, playerY: int) -> bool:
      return playerX < 0 or playerX == m or playerY < 0 or playerY == n \
          or grid[playerX][playerY] == '#'

    def canGoTo(playerX: int, playerY: int, fromX: int, fromY: int, boxX: int, boxY: int) -> bool:
      """Returns True if (playerX, playerY) can go to (fromX, fromY)."""
      q = collections.deque([(playerX, playerY)])
      seen = {(playerX, playerY)}

      while q:
        i, j = q.popleft()
        if i == fromX and j == fromY:
          return True
        for dx, dy in dirs:
          x = i + dx
          y = j + dy
          if isInvalid(x, y):
            continue
          if (x, y) in seen:
            continue
          if x == boxX and y == boxY:
            continue
          q.append((x, y))
          seen.add((x, y))

      return False

    ans = 0
    # (boxX, boxY, playerX, playerY)
    q = collections.deque([(box[0], box[1], player[0], player[1])])
    seen = {(box[0], box[1], player[0], player[1])}

    while q:
      for _ in range(len(q)):
        boxX, boxY, playerX, playerY = q.popleft()
        if boxX == target[0] and boxY == target[1]:
          return ans
        for dx, dy in dirs:
          nextBoxX = boxX + dx
          nextBoxY = boxY + dy
          if isInvalid(nextBoxX, nextBoxY):
            continue
          if (nextBoxX, nextBoxY, boxX, boxY) in seen:
            continue
          fromX = boxX + dirs[(k + 2) % 4]
          fromY = boxY + dirs[(k + 3) % 4]
          if isInvalid(fromX, fromY):
            continue
          if canGoTo(playerX, playerY, fromX, fromY, boxX, boxY):
            q.append((nextBoxX, nextBoxY, boxX, boxY))
            seen.add((nextBoxX, nextBoxY, boxX, boxY))
      ans += 1

    return -1


# Link: https://leetcode.com/problems/concatenated-words/description/
class Solution:
  def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    wordSet = set(words)

    @functools.lru_cache(None)
    def isConcat(word: str) -> bool:
      for i in range(1, len(word)):
        prefix = word[:i]
        suffix = word[i:]
        if prefix in wordSet and (suffix in wordSet or isConcat(suffix)):
          return True

      return False

    return [word for word in words if isConcat(word)]


# Link: https://leetcode.com/problems/replace-non-coprime-numbers-in-array/description/
class Solution:
  def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
    ans = []

    for num in nums:
      while ans and math.gcd(ans[-1], num) > 1:
        num = math.lcm(ans.pop(), num)
      ans.append(num)

    return ans


# Link: https://leetcode.com/problems/number-of-ships-in-a-rectangle/description/
# """
# This is Sea's API interface.
# You should not implement it, or speculate about its implementation
# """
# class Sea(object):
#   def hasShips(self, topRight: 'Point', bottomLeft: 'Point') -> bool:
#     pass
#
# class Point(object):
# def __init__(self, x: int, y: int):
# self.x = x
# self.y = y

class Solution(object):
  def countShips(self, sea: 'Sea', topRight: 'Point', bottomLeft: 'Point') -> int:
    if topRight.x < bottomLeft.x or topRight.y < bottomLeft.y:
      return 0
    if not sea.hasShips(topRight, bottomLeft):
      return 0

    # sea.hashShips(topRight, bottomLeft) == True
    if topRight.x == bottomLeft.x and topRight.y == bottomLeft.y:
      return 1

    mx = (topRight.x + bottomLeft.x) // 2
    my = (topRight.y + bottomLeft.y) // 2
    ans = 0
    # the top-right
    ans += self.countShips(sea, topRight, Point(mx + 1, my + 1))
    # the bottom-right
    ans += self.countShips(sea, Point(topRight.x, my),
                           Point(mx + 1, bottomLeft.y))
    # the top-left
    ans += self.countShips(sea, Point(mx, topRight.y),
                           Point(bottomLeft.x, my + 1))
    # the bottom-left
    ans += self.countShips(sea, Point(mx, my), bottomLeft)
    return ans


# Link: https://leetcode.com/problems/distance-to-a-cycle-in-undirected-graph/description/
class Solution:
  def distanceToCycle(self, n: int, edges: List[List[int]]) -> List[int]:
    ans = [0] * n
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    NO_RANK = -2

    # The minRank that u can reach with forward edges
    def getRank(u: int, currRank: int, rank: List[int]) -> int:
      if rank[u] != NO_RANK:  # The rank is already determined
        return rank[u]

      rank[u] = currRank
      minRank = currRank

      for v in graph[u]:
        # Visited or parent (that's why NO_RANK = -2 instead of -1)
        if rank[v] == len(rank) or rank[v] == currRank - 1:
          continue
        nextRank = getRank(v, currRank + 1, rank)
        # NextRank should > currRank if there's no cycle
        if nextRank <= currRank:
          cycle.append(v)
        minRank = min(minRank, nextRank)

      rank[u] = len(rank)  # Mark as visited.
      return minRank

    # rank[i] := the minimum node that node i can reach with forward edges
    # Initialize with NO_RANK = -2 to indicate not visited.
    cycle = []
    getRank(0, 0, [NO_RANK] * n)

    q = collections.deque(cycle)
    seen = set(cycle)

    dist = 0
    while q:
      dist += 1
      for _ in range(len(q)):
        u = q.popleft()
        for v in graph[u]:
          if v in seen:
            continue
          q.append(v)
          seen.add(v)
          ans[v] = dist

    return ans


# Link: https://leetcode.com/problems/minimum-deletions-to-make-array-divisible/description/
class Solution:
  def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
    gcd = functools.reduce(math.gcd, numsDivide)

    for i, num in enumerate(sorted(nums)):
      if gcd % num == 0:
        return i

    return -1


# Link: https://leetcode.com/problems/meeting-rooms-iii/description/
class Solution:
  def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
    count = [0] * n

    meetings.sort()

    occupied = []  # (endTime, roomId)
    availableRoomIds = [i for i in range(n)]
    heapq.heapify(availableRoomIds)

    for start, end in meetings:
      # Push meetings ending before this `meeting` in occupied to the
      # `availableRoomsIds`.
      while occupied and occupied[0][0] <= start:
        heapq.heappush(availableRoomIds, heapq.heappop(occupied)[1])
      if availableRoomIds:
        roomId = heapq.heappop(availableRoomIds)
        count[roomId] += 1
        heapq.heappush(occupied, (end, roomId))
      else:
        newStart, roomId = heapq.heappop(occupied)
        count[roomId] += 1
        heapq.heappush(occupied, (newStart + (end - start), roomId))

    return count.index(max(count))


# Link: https://leetcode.com/problems/random-pick-with-blacklist/description/
class Solution:
  def __init__(self, n: int, blacklist: List[int]):
    self.validRange = n - len(blacklist)
    self.dict = {}

    maxAvailable = n - 1

    for b in blacklist:
      self.dict[b] = -1

    for b in blacklist:
      if b < self.validRange:
        # Find the slot that haven't been used.
        while maxAvailable in self.dict:
          maxAvailable -= 1
        self.dict[b] = maxAvailable
        maxAvailable -= 1

  def pick(self) -> int:
    value = random.randint(0, self.validRange - 1)

    if value in self.dict:
      return self.dict[value]

    return value


# Link: https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/description/
class RandomizedCollection:
  def __init__(self):
    """
    Initialize your data structure here.
    """
    self.vals = []
    self.valToIndices = collections.defaultdict(list)

  def insert(self, val: int) -> bool:
    """
    Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
    """
    self.valToIndices[val].append(len(self.vals))
    self.vals.append([val, len(self.valToIndices[val]) - 1])
    return len(self.valToIndices[val]) == 1

  def remove(self, val: int) -> bool:
    """
    Removes a value from the collection. Returns true if the collection contained the specified element.
    """
    if val not in self.valToIndices or self.valToIndices[val] == []:
      return False

    index = self.valToIndices[val][-1]
    self.valToIndices[self.vals[-1][0]][self.vals[-1][1]] = index
    self.valToIndices[val].pop()
    self.vals[index] = self.vals[-1]
    self.vals.pop()
    return True

  def getRandom(self) -> int:
    """
    Get a random element from the collection.
    """
    index = random.randint(0, len(self.vals) - 1)
    return self.vals[index][0]


# Link: https://leetcode.com/problems/minimum-operations-to-form-subsequence-with-target-sum/description/
class Solution:
  def minOperations(self, nums: List[int], target: int) -> int:
    kNoMissingBit = 31
    maxBit = 31
    ans = 0
    minMissingBit = kNoMissingBit
    # count[i] := the number of occurrences of 2^i
    count = collections.Counter(int(math.log2(num)) for num in nums)

    for bit in range(maxBit):
      # Check if `bit` is in the target.
      if target >> bit & 1:
        # If there are available bits, use one bit.
        if count[bit] > 0:
          count[bit] -= 1
        else:
          minMissingBit = min(minMissingBit, bit)
      # If we previously missed a bit and there are available bits.
      if minMissingBit != kNoMissingBit and count[bit] > 0:
        count[bit] -= 1
        # Count the operations to break `bit` into `minMissingBit`.
        ans += bit - minMissingBit
        minMissingBit = kNoMissingBit  # Set it to an the invalid value.
      # Combining smaller numbers costs nothing.
      count[bit + 1] += count[bit] // 2

    # Check if all target bits have been covered, otherwise return -1.
    return ans if minMissingBit == maxBit else -1


# Link: https://leetcode.com/problems/paint-house-ii/description/
class Solution:
  def minCostII(self, costs: List[List[int]]) -> int:
    prevIndex = -1  # the previous minimum index
    prevMin1 = 0  # the minimum cost so far
    prevMin2 = 0  # the second minimum cost so far

    for cost in costs:  # O(n)
      # the painted index that will achieve the minimum cost after painting the
      # current house
      index = -1
      # the minimum cost after painting the current house
      min1 = math.inf
      # the second minimum cost after painting the current house
      min2 = math.inf
      for i, cst in enumerate(cost):   # O(k)
        theCost = cst + (prevMin2 if i == prevIndex else prevMin1)
        if theCost < min1:
          index = i
          min2 = min1
          min1 = theCost
        elif theCost < min2:  # min1 <= theCost < min2
          min2 = theCost

      prevIndex = index
      prevMin1 = min1
      prevMin2 = min2

    return prevMin1


# Link: https://leetcode.com/problems/the-maze-iii/description/
class Solution:
  def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
    ans = "impossible"
    minSteps = math.inf

    def dfs(i: int, j: int, dx: int, dy: int, steps: int, path: str):
      nonlocal ans
      nonlocal minSteps
      if steps >= minSteps:
        return

      if dx != 0 or dy != 0:  # Both are zeros for the initial ball position.
        while 0 <= i + dx < len(maze) and 0 <= j + dy < len(maze[0]) \
                and maze[i + dx][j + dy] != 1:
          i += dx
          j += dy
          steps += 1
          if i == hole[0] and j == hole[1] and steps < minSteps:
            minSteps = steps
            ans = path

      if maze[i][j] == 0 or steps + 2 < maze[i][j]:
        maze[i][j] = steps + 2  # +2 because maze[i][j] == 0 || 1.
        if dx == 0:
          dfs(i, j, 1, 0, steps, path + 'd')
        if dy == 0:
          dfs(i, j, 0, -1, steps, path + 'l')
        if dy == 0:
          dfs(i, j, 0, 1, steps, path + 'r')
        if dx == 0:
          dfs(i, j, -1, 0, steps, path + 'u')

    dfs(ball[0], ball[1], 0, 0, 0, '')
    return ans


# Link: https://leetcode.com/problems/largest-multiple-of-three/description/
class Solution:
  def largestMultipleOfThree(self, digits: List[int]) -> str:
    ans = ''
    mod1 = [1, 4, 7, 2, 5, 8]
    mod2 = [2, 5, 8, 1, 4, 7]
    count = collections.Counter(digits)
    summ = sum(digits)

    while summ % 3 != 0:
      for digit in (mod1 if summ % 3 == 1 else mod2):
        if count[digit]:
          count[digit] -= 1
          summ -= digit
          break

    for digit in reversed(range(10)):
      ans += str(digit) * count[digit]

    return '0' if len(ans) and ans[0] == '0' else ans


# Link: https://leetcode.com/problems/apply-operations-to-maximize-score/description/
class Solution:
  def maximumScore(self, nums: List[int], k: int) -> int:
    kMod = 1_000_000_007
    n = len(nums)
    ans = 1
    minPrimeFactors = self._sieveEratosthenes(max(nums) + 1)
    primeScores = [self._getPrimeScore(num, minPrimeFactors) for num in nums]
    # left[i] := the next index on the left (if any)
    #            s.t. primeScores[left[i]] >= primeScores[i]
    left = [-1] * n
    # right[i] := the next index on the right (if any)
    #             s.t. primeScores[right[i]] > primeScores[i]
    right = [n] * n
    stack = []

    # Find the next indices on the left where `primeScores` are greater or equal.
    for i in reversed(range(n)):
      while stack and primeScores[stack[-1]] <= primeScores[i]:
        left[stack.pop()] = i
      stack.append(i)

    stack = []

    # Find the next indices on the right where `primeScores` are greater.
    for i in range(n):
      while stack and primeScores[stack[-1]] < primeScores[i]:
        right[stack.pop()] = i
      stack.append(i)

    numAndIndexes = [(num, i) for i, num in enumerate(nums)]

    def modPow(x: int, n: int) -> int:
      if n == 0:
        return 1
      if n & 1:
        return x * modPow(x, n - 1) % kMod
      return modPow(x * x % kMod, n // 2)

    for num, i in sorted(numAndIndexes, key=lambda x: (-x[0], x[1])):
      # nums[i] is the maximum value in the range [left[i] + 1, right[i] - 1]
      # So, there are (i - left[i]) * (right[i] - 1) ranges where nums[i] will
      # be chosen.
      rangeCount = (i - left[i]) * (right[i] - i)
      actualCount = min(rangeCount, k)
      k -= actualCount
      ans *= modPow(num, actualCount)
      ans %= kMod

    return ans

  def _sieveEratosthenes(self, n: int) -> List[int]:
    """Gets the minimum prime factor of i, where 2 <= i <= n."""
    minPrimeFactors = [i for i in range(n + 1)]
    for i in range(2, int(n**0.5) + 1):
      if minPrimeFactors[i] == i:  # `i` is prime.
        for j in range(i * i, n, i):
          minPrimeFactors[j] = min(minPrimeFactors[j], i)
    return minPrimeFactors

  def _getPrimeScore(self, num: int, minPrimeFactors: List[int]) -> int:
    primeFactors = set()
    while num > 1:
      divisor = minPrimeFactors[num]
      primeFactors.add(divisor)
      while num % divisor == 0:
        num //= divisor
    return len(primeFactors)


# Link: https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
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
  def countSmaller(self, nums: List[int]) -> List[int]:
    ans = []
    ranks = self._getRanks(nums)
    tree = FenwickTree(len(ranks))

    for num in reversed(nums):
      ans.append(tree.get(ranks[num] - 1))
      tree.update(ranks[num], 1)

    return ans[::-1]

  def _getRanks(self, nums: List[int]) -> Dict[int, int]:
    ranks = collections.Counter()
    rank = 0
    for num in sorted(set(nums)):
      rank += 1
      ranks[num] = rank
    return ranks


# Link: https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
class Item:
  def __init__(self, num: int = 0, index: int = 0):
    self.num = num
    self.index = index


class Solution:
  def countSmaller(self, nums: List[int]) -> List[int]:
    n = len(nums)
    ans = [0] * n
    items = [Item(num, i) for i, num in enumerate(nums)]

    self._mergeSort(items, 0, n - 1, ans)
    return ans

  def _mergeSort(self, items: List[Item], l: int, r: int, ans: List[int]) -> None:
    if l >= r:
      return

    m = (l + r) // 2
    self._mergeSort(items, l, m, ans)
    self._mergeSort(items, m + 1, r, ans)
    self._merge(items, l, m, r, ans)

  def _merge(self, items: List[Item], l: int, m: int, r: int, ans: List[int]) -> None:
    sorted = [Item()] * (r - l + 1)
    k = 0  # sorted's index
    i = l  # left's index
    j = m + 1  # right's index
    rightCount = 0  # the number of numbers < items[i].num

    while i <= m and j <= r:
      if items[i].num > items[j].num:
        rightCount += 1
        sorted[k] = items[j]
        k += 1
        j += 1
      else:
        ans[items[i].index] += rightCount
        sorted[k] = items[i]
        k += 1
        i += 1

    # Put the possible remaining left part into the sorted array.
    while i <= m:
      ans[items[i].index] += rightCount
      sorted[k] = items[i]
      k += 1
      i += 1

    # Put the possible remaining right part into the sorted array.
    while j <= r:
      sorted[k] = items[j]
      k += 1
      j += 1

    items[l:l + len(sorted)] = sorted


# Link: https://leetcode.com/problems/maximum-number-of-books-you-can-take/description/
class Solution:
  def maximumBooks(self, books: List[int]) -> int:
    # dp[i] := the maximum the number of books we can take from books[0..i] with taking all of
    # books[i]
    dp = [0] * len(books)
    stack = []  # the possible indices we can reach

    for i, book in enumerate(books):
      # We may take all of books[j], where books[j] < books[i] - (i - j).
      while stack and books[stack[-1]] >= book - (i - stack[-1]):
        stack.pop()
      # We can now take books[j + 1..i].
      j = stack[-1] if stack else -1
      lastPicked = book - (i - j) + 1
      if lastPicked > 1:
        # book + (book - 1) + ... + (book - (i - j) + 1)
        dp[i] = (book + lastPicked) * (i - j) // 2
      else:
        # 1 + 2 + ... + book
        dp[i] = book * (book + 1) // 2
      if j >= 0:
        dp[i] += dp[j]
      stack.append(i)

    return max(dp)


# Link: https://leetcode.com/problems/transform-to-chessboard/description/
class Solution:
  def movesToChessboard(self, board: List[List[int]]) -> int:
    n = len(board)

    if any(board[0][0] ^ board[i][0] ^ board[0][j] ^ board[i][j] for i in range(n) for j in range(n)):
      return -1

    rowSum = sum(board[0])
    colSum = sum(board[i][0] for i in range(n))

    if rowSum != n // 2 and rowSum != (n + 1) // 2:
      return -1
    if colSum != n // 2 and colSum != (n + 1) // 2:
      return -1

    rowSwaps = sum(board[i][0] == (i & 1) for i in range(n))
    colSwaps = sum(board[0][i] == (i & 1) for i in range(n))

    if n & 1:
      if rowSwaps & 1:
        rowSwaps = n - rowSwaps
      if colSwaps & 1:
        colSwaps = n - colSwaps
    else:
      rowSwaps = min(rowSwaps, n - rowSwaps)
      colSwaps = min(colSwaps, n - colSwaps)

    return (rowSwaps + colSwaps) // 2


# Link: https://leetcode.com/problems/guess-the-word/description/
# """
# This is Master's API interface.
# You should not implement it, or speculate about its implementation
# """
# Class Master:
#   def guess(self, word: str) -> int:

class Solution:
  def findSecretWord(self, wordlist: List[str], master: 'Master') -> None:
    def getMatches(s1: str, s2: str) -> int:
      matches = 0
      for c1, c2 in zip(s1, s2):
        if c1 == c2:
          matches += 1
      return matches

    for _ in range(10):
      guessedWord = wordlist[random.randint(0, len(wordlist) - 1)]
      matches = master.guess(guessedWord)
      if matches == 6:
        break
      wordlist = [
          word for word in wordlist
          if getMatches(guessedWord, word) == matches]


# Link: https://leetcode.com/problems/sum-of-prefix-scores-of-strings/description/
class TrieNode:
  def __init__(self):
    self.children: Dict[str, TrieNode] = {}
    self.count = 0


class Solution:
  def sumPrefixScores(self, words: List[str]) -> List[int]:
    root = TrieNode()

    def insert(word: str) -> None:
      node: TrieNode = root
      for c in word:
        node = node.children.setdefault(c, TrieNode())
        node.count += 1

    for word in words:
      insert(word)

    def getScore(word: str) -> int:
      node: TrieNode = root
      score = 0
      for c in word:
        node = node.children[c]
        score += node.count
      return score

    return [getScore(word) for word in words]


# Link: https://leetcode.com/problems/k-similar-strings/description/
class Solution:
  def kSimilarity(self, s1: str, s2: str) -> int:
    ans = 0
    q = collections.deque([s1])
    seen = {s1}

    while q:
      for _ in range(len(q)):
        curr = q.popleft()
        if curr == s2:
          return ans
        for child in self._getChildren(curr, s2):
          if child in seen:
            continue
          q.append(child)
          seen.add(child)
      ans += 1

    return -1

  def _getChildren(self, curr: str, target: str) -> List[str]:
    children = []
    s = list(curr)
    i = 0  # the first index s.t. curr[i] != target[i]
    while curr[i] == target[i]:
      i += 1

    for j in range(i + 1, len(s)):
      if s[j] == target[i]:
        s[i], s[j] = s[j], s[i]
        children.append(''.join(s))
        s[i], s[j] = s[j], s[i]

    return children


# Link: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/description/
class Solution:
  def findMin(self, nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1

    while l < r:
      m = (l + r) // 2
      if nums[m] == nums[r]:
        r -= 1
      elif nums[m] < nums[r]:
        r = m
      else:
        l = m + 1

    return nums[l]


# Link: https://leetcode.com/problems/check-if-string-is-transformable-with-substring-sort-operations/description/
class Solution:
  def isTransformable(self, s: str, t: str) -> bool:
    if collections.Counter(s) != collections.Counter(t):
      return False

    positions = [collections.deque() for _ in range(10)]

    for i, c in enumerate(s):
      positions[int(c)].append(i)

    # For each digit in `t`, check if we can put this digit in `s` at the same
    # position as `t`. Ensure that all the left digits are equal to or greater
    # than it. This is because the only operation we can perform is sorting in
    # ascending order. If there is a digit to the left that is smaller than it,
    # we can never move it to the same position as in `t`. However, if all the
    # digits to its left are equal to or greater than it, we can move it one
    # position to the left until it reaches the same position as in `t`.
    for c in t:
      d = int(c)
      front = positions[d].popleft()
      for smaller in range(d):
        if positions[smaller] and positions[smaller][0] < front:
          return False

    return True


# Link: https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/description/
class Solution:
  def numOfArrays(self, n: int, m: int, k: int) -> int:
    kMod = 1_000_000_007
    # dp[i][j][k] := the number of ways to build an array of length i, where j
    # is the maximum number and k is `search_cost`
    dp = [[[0] * (k + 1) for j in range(m + 1)] for _ in range(n + 1)]

    for j in range(1, m + 1):
      dp[1][j][1] = 1

    for i in range(2, n + 1):  # for each length
      for j in range(1, m + 1):  # for each max value
        for cost in range(1, k + 1):  # for each cost
          # 1. Appending any of [1, j] in the i-th position doesn't change the
          #    maximum and cost.
          dp[i][j][cost] = j * dp[i - 1][j][cost] % kMod
          # 2. Appending j in the i-th position makes j the new max and cost 1.
          for prevMax in range(1, j):
            dp[i][j][cost] += dp[i - 1][prevMax][cost - 1]
            dp[i][j][cost] %= kMod

    return sum(dp[n][j][k] for j in range(1, m + 1)) % kMod


# Link: https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/description/
class Solution:
  def numOfArrays(self, n: int, m: int, k: int) -> int:
    kMod = 1_000_000_007
    # dp[i][j][k] := the number of ways to build an array of length i, where j
    # is the maximum number and k is the `search_cost`
    dp = [[[0] * (k + 1) for j in range(m + 1)] for _ in range(n + 1)]
    # prefix[i][j][k] := sum(dp[i][x][k]), where 1 <= x <= j
    prefix = [[[0] * (k + 1) for j in range(m + 1)] for _ in range(n + 1)]

    for j in range(1, m + 1):
      dp[1][j][1] = 1
      prefix[1][j][1] = j

    for i in range(2, n + 1):  # for each length
      for j in range(1, m + 1):  # for each max value
        for cost in range(1, k + 1):  # for each cost
          # 1. Appending any of [1, j] in the i-th position doesn't change the
          #    maximum and cost.
          # 2. Appending j in the i-th position makes j the new max and cost 1.
          dp[i][j][cost] = (j * dp[i - 1][j][cost] +
                            prefix[i - 1][j - 1][cost - 1]) % kMod
          prefix[i][j][cost] = (dp[i][j][cost] + prefix[i][j - 1][cost]) % kMod

    return sum(dp[n][j][k] for j in range(1, m + 1)) % kMod


# Link: https://leetcode.com/problems/shortest-distance-from-all-buildings/description/
class Solution:
  def shortestDistance(self, grid: List[List[int]]) -> int:
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    m = len(grid)
    n = len(grid[0])
    nBuildings = sum(a == 1 for row in grid for a in row)
    ans = math.inf
    # dist[i][j] := the total distance of grid[i][j] (0) to reach all the
    # buildings (1)
    dist = [[0] * n for _ in range(m)]
    # reachCount[i][j] := the number of buildings (1) grid[i][j] (0) can reach
    reachCount = [[0] * n for _ in range(m)]

    def bfs(row: int, col: int) -> bool:
      q = collections.deque([(row, col)])
      seen = {(row, col)}
      depth = 0
      seenBuildings = 1

      while q:
        depth += 1
        for _ in range(len(q)):
          i, j = q.popleft()
          for dx, dy in dirs:
            x = i + dx
            y = j + dy
            if x < 0 or x == m or y < 0 or y == n:
              continue
            if (x, y) in seen:
              continue
            seen.add((x, y))
            if not grid[x][y]:
              dist[x][y] += depth
              reachCount[x][y] += 1
              q.append((x, y))
            elif grid[x][y] == 1:
              seenBuildings += 1

      # True if all the buildings (1) are connected
      return seenBuildings == nBuildings

    for i in range(m):
      for j in range(n):
        if grid[i][j] == 1:  # BFS from this building.
          if not bfs(i, j):
            return -1

    for i in range(m):
      for j in range(n):
        if reachCount[i][j] == nBuildings:
          ans = min(ans, dist[i][j])

    return -1 if ans == math.inf else ans


# Link: https://leetcode.com/problems/rearrange-string-k-distance-apart/description/
class Solution:
  def rearrangeString(self, s: str, k: int) -> str:
    n = len(s)
    ans = []
    count = collections.Counter(s)
    # valid[i] := the leftmost index i can appear
    valid = collections.Counter()

    def getBestLetter(index: int) -> chr:
      """Returns the valid letter that has the most count."""
      maxCount = -1
      bestLetter = '*'

      for c in string.ascii_lowercase:
        if count[c] > 0 and count[c] > maxCount and index >= valid[c]:
          bestLetter = c
          maxCount = count[c]

      return bestLetter

    for i in range(n):
      c = getBestLetter(i)
      if c == '*':
        return ''
      ans.append(c)
      count[c] -= 1
      valid[c] = i + k

    return ''.join(ans)


# Link: https://leetcode.com/problems/count-ways-to-make-array-with-product/description/
class Solution:
  def waysToFillArray(self, queries: List[List[int]]) -> List[int]:
    kMod = 1_000_000_007
    kMax = 10_000
    minPrimeFactors = self._sieveEratosthenes(kMax + 1)

    @functools.lru_cache(None)
    def fact(i: int) -> int:
      return 1 if i <= 1 else i * fact(i - 1) % kMod

    @functools.lru_cache(None)
    def inv(i: int) -> int:
      return pow(i, kMod - 2, kMod)

    @functools.lru_cache(None)
    def nCk(n: int, k: int) -> int:
      return fact(n) * inv(fact(k)) * inv(fact(n - k)) % kMod

    ans = []

    for n, k in queries:
      res = 1
      for freq in self._getPrimeFactorsCount(k, minPrimeFactors).values():
        res = res * nCk(n - 1 + freq, freq) % kMod
      ans.append(res)

    return ans

  def _sieveEratosthenes(self, n: int) -> List[int]:
    """Gets the minimum prime factor of i, where 1 < i <= n."""
    minPrimeFactors = [i for i in range(n + 1)]
    for i in range(2, int(n**0.5) + 1):
      if minPrimeFactors[i] == i:  # `i` is prime.
        for j in range(i * i, n, i):
          minPrimeFactors[j] = min(minPrimeFactors[j], i)
    return minPrimeFactors

  def _getPrimeFactorsCount(self, num: int, minPrimeFactors: List[int]) -> Dict[int, int]:
    count = collections.Counter()
    while num > 1:
      divisor = minPrimeFactors[num]
      while num % divisor == 0:
        num //= divisor
        count[divisor] += 1
    return count


# Link: https://leetcode.com/problems/shortest-path-visiting-all-nodes/description/
class Solution:
  def shortestPathLength(self, graph: List[List[int]]) -> int:
    n = len(graph)
    goal = (1 << n) - 1

    ans = 0
    q = collections.deque()  # (u, state)
    seen = set()

    for i in range(n):
      q.append((i, 1 << i))

    while q:
      for _ in range(len(q)):
        u, state = q.popleft()
        if state == goal:
          return ans
        if (u, state) in seen:
          continue
        seen.add((u, state))
        for v in graph[u]:
          q.append((v, state | (1 << v)))
      ans += 1

    return -1


# Link: https://leetcode.com/problems/smallest-good-base/description/
class Solution:
  def smallestGoodBase(self, n: str) -> str:
    n = int(n)

    for m in range(int(math.log(n, 2)), 1, -1):
      k = int(n**m**-1)
      if (k**(m + 1) - 1) // (k - 1) == n:
        return str(k)

    return str(n - 1)


# Link: https://leetcode.com/problems/power-of-heroes/description/
class Solution:
  def sumOfPower(self, nums: List[int]) -> int:
    kMod = 1_000_000_007
    ans = 0
    summ = 0

    for num in sorted(nums):
      ans += (num + summ) * num**2
      ans %= kMod
      summ = (summ * 2 + num) % kMod

    return ans


# Link: https://leetcode.com/problems/longest-cycle-in-a-graph/description/
class Solution:
  def longestCycle(self, edges: List[int]) -> int:
    ans = -1
    time = 1
    timeVisited = [0] * len(edges)

    for i, edge in enumerate(edges):
      if timeVisited[i]:
        continue
      startTime = time
      u = i
      while u != -1 and not timeVisited[u]:
        timeVisited[u] = time
        time += 1
        u = edges[u]  # Move to the next node.
      if u != -1 and timeVisited[u] >= startTime:
        ans = max(ans, time - timeVisited[u])

    return ans


# Link: https://leetcode.com/problems/the-score-of-students-solving-math-expression/description/
class Solution:
  def scoreOfStudents(self, s: str, answers: List[int]) -> int:
    n = len(s) // 2 + 1
    ans = 0
    func = {'+': operator.add, '*': operator.mul}
    dp = [[set() for j in range(n)] for _ in range(n)]

    for i in range(n):
      dp[i][i].add(int(s[i * 2]))

    for d in range(1, n):
      for i in range(n - d):
        j = i + d
        for k in range(i, j):
          op = s[k * 2 + 1]
          for a in dp[i][k]:
            for b in dp[k + 1][j]:
              res = func[op](a, b)
              if res <= 1000:
                dp[i][j].add(res)

    correctAnswer = eval(s)

    for answer, freq in collections.Counter(answers).items():
      if answer == correctAnswer:
        ans += 5 * freq
      elif answer in dp[0][n - 1]:
        ans += 2 * freq

    return ans


# Link: https://leetcode.com/problems/frog-jump/description/
class Solution:
  def canCross(self, stones: List[int]) -> bool:
    n = len(stones)
    # dp[i][j] := True if a frog can make a size j jump from stones[i]
    dp = [[False] * (n + 1) for _ in range(n)]
    dp[0][1] = True

    for i in range(1, n):
      for j in range(i):
        k = stones[i] - stones[j]
        if k <= n and dp[j][k]:
          dp[i][k - 1] = True
          dp[i][k] = True
          dp[i][k + 1] = True

    return any(dp[-1])


# Link: https://leetcode.com/problems/frog-jump/description/
class Solution:
  def canCross(self, stones: List[int]) -> bool:
    n = len(stones)
    # dp[i][j] := True if a frog can make a size j jump to stones[i]
    dp = [[False] * (n + 1) for _ in range(n)]
    dp[0][0] = True

    for i in range(1, n):
      for j in range(i):
        k = stones[i] - stones[j]
        if k > n:
          continue
        for x in (k - 1, k, k + 1):
          if 0 <= x <= n:
            dp[i][k] |= dp[j][x]

    return any(dp[-1])


# Link: https://leetcode.com/problems/orderly-queue/description/
class Solution:
  def orderlyQueue(self, s: str, k: int) -> str:
    return ''.join(sorted(s)) if k > 1 \
        else min(s[i:] + s[:i] for i in range(len(s)))


# Link: https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/
class Solution:
  def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    jobs = sorted([(s, e, p) for s, e, p in zip(startTime, endTime, profit)])

    # Will use binary search to find the first available startTime
    for i in range(len(startTime)):
      startTime[i] = jobs[i][0]

    @functools.lru_cache(None)
    def dp(i: int) -> int:
      """Returns the maximum profit to schedule jobs[i..n)."""
      if i == len(startTime):
        return 0
      j = bisect.bisect_left(startTime, jobs[i][1])
      return max(jobs[i][2] + dp(j), dp(i + 1))

    return dp(0)


# Link: https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/
class Solution:
  def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    # dp[i] := the maximum profit to schedule jobs[i..n)
    dp = [0] * (len(startTime) + 1)
    jobs = sorted([(s, e, p) for s, e, p in zip(startTime, endTime, profit)])

    for i in range(len(startTime)):
      startTime[i] = jobs[i][0]

    for i in reversed(range(len(startTime))):
      j = bisect.bisect_left(startTime, jobs[i][1])
      dp[i] = max(jobs[i][2] + dp[j], dp[i + 1])

    return dp[0]


# Link: https://leetcode.com/problems/maximum-profit-in-job-scheduling/description/
class Solution:
  def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    maxProfit = 0
    jobs = sorted([(s, e, p) for s, e, p in zip(startTime, endTime, profit)])
    minHeap = []  # (endTime, profit)

    # Will use binary search to find the first available startTime
    for i in range(len(startTime)):
      startTime[i] = jobs[i][0]

    for s, e, p in jobs:
      while minHeap and s >= minHeap[0][0]:
        maxProfit = max(maxProfit, heapq.heappop(minHeap)[1])
      heapq.heappush(minHeap, (e, p + maxProfit))

    return max(maxProfit, max(p for _, p in minHeap))


# Link: https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/description/
class Solution:
  def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
    @functools.lru_cache(2000)
    def dp(s: int, i: int) -> int:
      """Returns the maximum score of nums[s..e] and multipliers[i]."""
      if i == len(multipliers):
        return 0

      # The number of nums picked on the start side is s.
      # The number of nums picked on the end side is i - s.
      # So, e = n - (i - s) - 1.
      e = len(nums) - (i - s) - 1
      pickStart = nums[s] * multipliers[i] + dp(s + 1, i + 1)
      pickEnd = nums[e] * multipliers[i] + dp(s, i + 1)
      return max(pickStart, pickEnd)

    return dp(0, 0)


# Link: https://leetcode.com/problems/optimize-water-distribution-in-a-village/description/
class Solution:
  def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
    ans = 0
    graph = [[] for _ in range(n + 1)]
    minHeap = []  # (d, u)

    for u, v, w in pipes:
      graph[u].append((v, w))
      graph[v].append((u, w))

    # Connect virtual 0 with nodes 1 to n.
    for i, well in enumerate(wells):
      graph[0].append((i + 1, well))
      heapq.heappush(minHeap, (well, i + 1))

    mst = {0}

    while len(mst) < n + 1:
      d, u = heapq.heappop(minHeap)
      if u in mst:
        continue
      # Add the new vertex.
      mst.add(u)
      ans += d
      # Expand if possible.
      for v, w in graph[u]:
        if v not in mst:
          heapq.heappush(minHeap, (w, v))

    return ans


# Link: https://leetcode.com/problems/parallel-courses-iii/description/
class Solution:
  def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
    graph = [[] for _ in range(n)]
    inDegrees = [0] * n
    dist = time.copy()

    # Build the graph.
    for a, b in relations:
      u = a - 1
      v = b - 1
      graph[u].append(v)
      inDegrees[v] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    while q:
      u = q.popleft()
      for v in graph[u]:
        dist[v] = max(dist[v], dist[u] + time[v])
        inDegrees[v] -= 1
        if inDegrees[v] == 0:
          q.append(v)

    return max(dist)


# Link: https://leetcode.com/problems/cracking-the-safe/description/
class Solution:
  def crackSafe(self, n: int, k: int) -> str:
    passwordSize = k**n
    path = '0' * n
    seen = set()
    seen.add(path)

    def dfs(path: str) -> str:
      if len(seen) == passwordSize:
        return path

      for c in map(str, range(k)):
        node = path[-n + 1:] + c if n > 1 else c
        if node not in seen:
          seen.add(node)
          res = dfs(path + c)
          if res:
            return res
          seen.remove(node)

    return dfs(path)


# Link: https://leetcode.com/problems/kth-smallest-instructions/description/
class Solution:
  def kthSmallestPath(self, destination: List[int], k: int) -> str:
    ans = []
    v, h = destination

    for _ in range(h + v):
      # If pick 'H', then we're able to reack 1, 2, ..., availableRank.
      availableRank = math.comb(h + v - 1, v)
      if availableRank >= k:  # Should pick 'H'.
        ans.append('H')
        h -= 1
      else:  # Should pick 'V'.
        k -= availableRank
        ans.append('V')
        v -= 1

    return ''.join(ans)


# Link: https://leetcode.com/problems/longest-subsequence-repeated-k-times/description/
class Solution:
  def longestSubsequenceRepeatedK(self, s: str, k: int) -> str:
    ans = ''
    count = [0] * 26
    possibleChars = []
    # Stores subsequences, where the length grows by 1 each time.
    q = collections.deque([''])

    for c in s:
      count[ord(c) - ord('a')] += 1

    for c in string.ascii_lowercase:
      if count[ord(c) - ord('a')] >= k:
        possibleChars.append(c)

    def isSubsequence(subseq: str, s: str, k: int) -> bool:
      i = 0  # subseq's index
      for c in s:
        if c == subseq[i]:
          i += 1
          if i == len(subseq):
            k -= 1
            if k == 0:
              return True
            i = 0
      return False

    while q:
      currSubseq = q.popleft()
      if len(currSubseq) * k > len(s):
        return ans
      for c in possibleChars:
        newSubseq = currSubseq + c
        if isSubsequence(newSubseq, s, k):
          q.append(newSubseq)
          ans = newSubseq

    return ans


# Link: https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/description/
class Solution:
  def minMovesToMakePalindrome(self, s: str) -> int:
    ans = 0
    chars = list(s)

    while len(chars) > 1:
      # Greedily match the last digit.
      i = chars.index(chars[-1])
      if i == len(chars) - 1:
        # s[i] is the middle letter.
        ans += i // 2
      else:
        chars.pop(i)
        ans += i  # Swap the matched letter to the left.
      chars.pop()

    return ans


# Link: https://leetcode.com/problems/design-movie-rental-system/description/
from sortedcontainers import SortedList


class MovieRentingSystem:
  def __init__(self, n: int, entries: List[List[int]]):
    self.unrented = collections.defaultdict(
        SortedList)  # {movie: (price, shop)}
    self.shopAndMovieToPrice = {}  # {(shop, movie): price}
    self.rented = SortedList()  # (price, shop, movie)
    for shop, movie, price in entries:
      self.unrented[movie].add((price, shop))
      self.shopAndMovieToPrice[(shop, movie)] = price

  def search(self, movie: int) -> List[int]:
    return [shop for _, shop in self.unrented[movie][:5]]

  def rent(self, shop: int, movie: int) -> None:
    price = self.shopAndMovieToPrice[(shop, movie)]
    self.unrented[movie].remove((price, shop))
    self.rented.add((price, shop, movie))

  def drop(self, shop: int, movie: int) -> None:
    price = self.shopAndMovieToPrice[(shop, movie)]
    self.unrented[movie].add((price, shop))
    self.rented.remove((price, shop, movie))

  def report(self) -> List[List[int]]:
    return [[shop, movie] for _, shop, movie in self.rented[:5]]


# Link: https://leetcode.com/problems/minimum-number-of-days-to-eat-n-oranges/description/
class Solution:
  @functools.lru_cache(None)
  def minDays(self, n: int) -> int:
    if n <= 1:
      return n
    return 1 + min(self.minDays(n // 3) + n % 3,
                   self.minDays(n // 2) + n % 2)


# Link: https://leetcode.com/problems/constrained-subsequence-sum/description/
class Solution:
  def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
    # dp[i] := the maximum the sum of non-empty subsequences in nums[0..i]
    dp = [0] * len(nums)
    # dq stores dp[i - k], dp[i - k + 1], ..., dp[i - 1] whose values are > 0
    # in decreasing order.
    dq = collections.deque()

    for i, num in enumerate(nums):
      if dq:
        dp[i] = max(dq[0], 0) + num
      else:
        dp[i] = num
      while dq and dq[-1] < dp[i]:
        dq.pop()
      dq.append(dp[i])
      if i >= k and dp[i - k] == dq[0]:
        dq.popleft()

    return max(dp)


# Link: https://leetcode.com/problems/find-the-closest-palindrome/description/
class Solution:
  def nearestPalindromic(self, n: str) -> str:
    def getPalindromes(s: str) -> tuple:
      num = int(s)
      k = len(s)
      palindromes = []
      half = s[0:(k + 1) // 2]
      reversedHalf = half[:k // 2][::-1]
      candidate = int(half + reversedHalf)

      if candidate < num:
        palindromes.append(candidate)
      else:
        prevHalf = str(int(half) - 1)
        reversedPrevHalf = prevHalf[:k // 2][::-1]
        if k % 2 == 0 and int(prevHalf) == 0:
          palindromes.append(9)
        elif k % 2 == 0 and (int(prevHalf) + 1) % 10 == 0:
          palindromes.append(int(prevHalf + '9' + reversedPrevHalf))
        else:
          palindromes.append(int(prevHalf + reversedPrevHalf))

      if candidate > num:
        palindromes.append(candidate)
      else:
        nextHalf = str(int(half) + 1)
        reversedNextHalf = nextHalf[:k // 2][::-1]
        palindromes.append(int(nextHalf + reversedNextHalf))

      return palindromes

    prevPalindrome, nextPalindrome = getPalindromes(n)
    return str(prevPalindrome) if abs(prevPalindrome - int(n)) <= abs(nextPalindrome - int(n)) else str(nextPalindrome)


# Link: https://leetcode.com/problems/number-of-excellent-pairs/description/
class Solution:
  def countExcellentPairs(self, nums: List[int], k: int) -> int:
    count = collections.Counter(map(int.bit_count, set(nums)))
    return sum(count[i] * count[j]
               for i in count
               for j in count
               if i + j >= k)


# Link: https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/description/
from enum import IntEnum


class Pos(IntEnum):
  kHorizontal = 0
  kVertical = 1


class Solution:
  def minimumMoves(self, grid: List[List[int]]) -> int:
    n = len(grid)
    ans = 0
    # the state of (x, y, pos)
    # pos := 0 (horizontal) / 1 (vertical)
    q = collections.deque([(0, 0, Pos.kHorizontal)])
    seen = {(0, 0, Pos.kHorizontal)}

    def canMoveRight(x: int, y: int, pos: Pos) -> bool:
      if pos == Pos.kHorizontal:
        return y + 2 < n and not grid[x][y + 2]
      return y + 1 < n and not grid[x][y + 1] and not grid[x + 1][y + 1]

    def canMoveDown(x: int, y: int, pos: Pos) -> bool:
      if pos == Pos.kVertical:
        return x + 2 < n and not grid[x + 2][y]
      return x + 1 < n and not grid[x + 1][y] and not grid[x + 1][y + 1]

    def canRotateClockwise(x: int, y: int, pos: Pos) -> bool:
      return pos == Pos.kHorizontal and x + 1 < n and \
          not grid[x + 1][y + 1] and not grid[x + 1][y]

    def canRotateCounterclockwise(x: int, y: int, pos: Pos) -> bool:
      return pos == Pos.kVertical and y + 1 < n and \
          not grid[x + 1][y + 1] and not grid[x][y + 1]

    while q:
      for _ in range(len(q)):
        x, y, pos = q.popleft()
        if x == n - 1 and y == n - 2 and pos == Pos.kHorizontal:
          return ans
        if canMoveRight(x, y, pos) and (x, y + 1, pos) not in seen:
          q.append((x, y + 1, pos))
          seen.add((x, y + 1, pos))
        if canMoveDown(x, y, pos) and (x + 1, y, pos) not in seen:
          q.append((x + 1, y, pos))
          seen.add((x + 1, y, pos))
        newPos = Pos.kVertical if pos == Pos.kHorizontal else Pos.kHorizontal
        if (canRotateClockwise(x, y, pos) or canRotateCounterclockwise(x, y, pos)) and \
                (x, y, newPos) not in seen:
          q.append((x, y, newPos))
          seen.add((x, y, newPos))
      ans += 1

    return -1


# Link: https://leetcode.com/problems/cycle-length-queries-in-a-tree/description/
class Solution:
  def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
    def getCycleLength(a: int, b: int):
      cycleLength = 1
      while a != b:
        if a > b:
          a //= 2
        else:
          b //= 2
        cycleLength += 1
      return cycleLength

    return [getCycleLength(*query) for query in queries]


# Link: https://leetcode.com/problems/number-of-valid-subarrays/description/
class Solution:
  def validSubarrays(self, nums: List[int]) -> int:
    # For each `num` in `nums`, each element x in the stack can be the leftmost
    # element s.t. [x, num] forms a valid subarray, so the size of the stack is
    # the number of valid subarrays ending in the current number.
    #
    # e.g. nums = [1, 3, 2]
    # num = 1, stack = [1] -> valid subarray is [1]
    # num = 3, stack = [1, 3] -> valid subarrays are [1, 3], [3]
    # num = 2, stack = [1, 2] -> valid subarrays are [1, 3, 2], [2]
    ans = 0
    stack = []

    for num in nums:
      while stack and stack[-1] > num:
        stack.pop()
      stack.append(num)
      ans += len(stack)

    return ans


# Link: https://leetcode.com/problems/maximum-score-of-a-good-subarray/description/
class Solution:
  # Similar to 84. Largest Rectangle in Histogram
  def maximumScore(self, nums: List[int], k: int) -> int:
    ans = 0
    stack = []

    for i in range(len(nums) + 1):
      while stack and (i == len(nums) or nums[stack[-1]] > nums[i]):
        h = nums[stack.pop()]
        w = i - stack[-1] - 1 if stack else i
        if (not stack or stack[-1] + 1 <= k) and i - 1 >= k:
          ans = max(ans, h * w)
      stack.append(i)

    return ans


# Link: https://leetcode.com/problems/maximum-score-of-a-good-subarray/description/
class Solution:
  def maximumScore(self, nums: List[int], k: int) -> int:
    n = len(nums)
    ans = nums[k]
    mini = nums[k]
    i = k
    j = k

    # Greedily expand the window and decrease the minimum as slow as possible.
    while i > 0 or j < n - 1:
      if i == 0:
        j += 1
      elif j == n - 1:
        i -= 1
      elif nums[i - 1] < nums[j + 1]:
        j += 1
      else:
        i -= 1
      mini = min(mini, nums[i], nums[j])
      ans = max(ans, mini * (j - i + 1))

    return ans


# Link: https://leetcode.com/problems/make-array-strictly-increasing/description/
class Solution:
  def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
    # dp[i] := the minimum steps to reach i at previous round
    dp = {-1: 0}

    arr2.sort()

    for a in arr1:
      nextDp = collections.defaultdict(lambda: math.inf)
      for val, steps in dp.items():
        # It's possible to use the value in the arr1.
        if a > val:
          nextDp[a] = min(nextDp[a], steps)
        # Also try the value in the arr2.
        i = bisect_right(arr2, val)
        if i < len(arr2):
          nextDp[arr2[i]] = min(nextDp[arr2[i]], steps + 1)
      if not nextDp:
        return -1
      dp = nextDp

    return min(dp.values())


# Link: https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/description/
class Solution:
  def minimumDifference(self, nums: List[int]) -> int:
    n = len(nums) // 3
    ans = math.inf
    leftSum = 0
    rightSum = 0
    maxHeap = []  # Left part, as small as possible
    minHeap = []  # Right part, as big as possible
    # minLeftSum[i] := the minimum of the sum of n nums in nums[0..i)
    minLeftSum = [0] * len(nums)

    for i in range(2 * n):
      heapq.heappush(maxHeap, -nums[i])
      leftSum += nums[i]
      if len(maxHeap) == n + 1:
        leftSum += heapq.heappop(maxHeap)
      if len(maxHeap) == n:
        minLeftSum[i] = leftSum

    for i in range(len(nums) - 1, n - 1, -1):
      heapq.heappush(minHeap, nums[i])
      rightSum += nums[i]
      if len(minHeap) == n + 1:
        rightSum -= heapq.heappop(minHeap)
      if len(minHeap) == n:
        ans = min(ans, minLeftSum[i - 1] - rightSum)

    return ans


# Link: https://leetcode.com/problems/count-visited-nodes-in-a-directed-graph/description/
class Solution:
  def countVisitedNodes(self, edges: List[int]) -> List[int]:
    n = len(edges)
    ans = [0] * n
    inDegrees = [0] * n
    seen = [False] * n
    stack = []

    for v in edges:
      inDegrees[v] += 1

    # Perform topological sorting.
    q = collections.deque([i for i, d in enumerate(inDegrees) if d == 0])

    # Push non-cyclic nodes to stack.
    while q:
      u = q.popleft()
      inDegrees[edges[u]] -= 1
      if inDegrees[edges[u]] == 0:
        q.append(edges[u])
      stack.append(u)
      seen[u] = True

    # Fill the length of cyclic nodes.
    for i in range(n):
      if not seen[i]:
        self._fillCycle(edges, i, seen, ans)

    # Fill the length of non-cyclic nodes.
    while stack:
      u = stack.pop()
      ans[u] = ans[edges[u]] + 1

    return ans

  def _fillCycle(self, edges: List[int], start: int, seen: List[bool], ans: List[int]) -> None:
    cycleLength = 0
    u = start
    while not seen[u]:
      cycleLength += 1
      seen[u] = True
      u = edges[u]
    ans[start] = cycleLength
    u = edges[start]
    while u != start:
      ans[u] = cycleLength
      u = edges[u]


# Link: https://leetcode.com/problems/maximum-elegance-of-a-k-length-subsequence/description/
class Solution:
  def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
    ans = 0
    totalProfit = 0
    seenCategories = set()
    decreasingDuplicateProfits = []

    items.sort(reverse=True)

    for i in range(k):
      profit, category = items[i]
      totalProfit += profit
      if category in seenCategories:
        decreasingDuplicateProfits.append(profit)
      else:
        seenCategories.add(category)

    ans = totalProfit + len(seenCategories)**2

    for i in range(k, len(items)):
      profit, category = items[i]
      if category not in seenCategories and decreasingDuplicateProfits:
        # If this is a new category we haven't seen before, it's worth
        # considering taking it and replacing the one with the least profit
        # since it will increase the distinct_categories and potentially result
        # in a larger total_profit + distinct_categories^2.
        totalProfit -= decreasingDuplicateProfits.pop()
        totalProfit += profit
        seenCategories.add(category)
        ans = max(ans, totalProfit + len(seenCategories)**2)

    return ans


# Link: https://leetcode.com/problems/split-message-based-on-limit/description/
class Solution:
  def splitMessage(self, message: str, limit: int) -> List[str]:
    kMessageLength = len(message)

    def sz(num: int):
      return len(str(num))

    b = 1
    # the total length of a: initialized with the length of "1"
    aLength = sz(1)

    # the total length of b := b * sz(b)
    # The total length of "</>" := b * 3
    while b * limit < b * (sz(b) + 3) + aLength + kMessageLength:
      # If the length of the last suffix "<b/b>" := sz(b) * 2 + 3 >= limit,
      # then it's impossible that the length of "*<b/b>" <= limit.
      if sz(b) * 2 + 3 >= limit:
        return []
      b += 1
      aLength += sz(b)

    ans = []

    i = 0
    for a in range(1, b + 1):
      # the length of "<a/b>" := sz(a) + sz(b) + 3
      j = limit - (sz(a) + sz(b) + 3)
      ans.append(f'{message[i:i + j]}<{a}/{b}>')
      i += j

    return ans


# Link: https://leetcode.com/problems/string-transformation/description/
class Solution:
  # This dynamic programming table dp[k][i] represents the number of ways to
  # rearrange the String s after k steps such that it starts with s[i].
  # A String can be rotated from 1 to n - 1 times. The transition rule is
  # dp[k][i] = sum(dp[k - 1][j]) for all j != i. For example, when n = 4 and
  # k = 3, the table looks like this:
  #
  # -----------------------------------------------------------
  # |       | i = 0 | i = 1 | i = 2 | i = 3 | sum = (n - 1)^k |
  # -----------------------------------------------------------
  # | k = 0 |   1   |   0   |   0   |   0   |        1        |
  # | k = 1 |   0   |   1   |   1   |   1   |        3        |
  # | k = 2 |   3   |   2   |   2   |   2   |        9        |
  # | k = 3 |   6   |   7   |   7   |   7   |       27        |
  # -----------------------------------------------------------
  #
  # By observation, we have
  #   * dp[k][!0] = ((n - 1)^k - (-1)^k) / n
  #   * dp[k][0] = dp[k][!0] + (-1)^k
  def numberOfWays(self, s: str, t: str, k: int) -> int:
    kMod = 1_000_000_007
    n = len(s)
    negOnePowK = 1 if k % 2 == 0 else -1  # (-1)^k
    z = self._zFunction(s + t + t)
    # indices in `s` s.t. for each `i` in the returned indices,
    # `s[i..n) + s[0..i) = t`.
    indices = [i - n for i in range(n, n + n) if z[i] >= n]
    dp = [0] * 2  # dp[0] := dp[k][0]; dp[1] := dp[k][!0]
    dp[1] = (pow(n - 1, k, kMod) - negOnePowK) * pow(n, kMod - 2, kMod)
    dp[0] = dp[1] + negOnePowK
    return sum(dp[0] if index == 0 else dp[1] for index in indices) % kMod

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


# Link: https://leetcode.com/problems/collect-coins-in-a-tree/description/
class Solution:
  def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
    n = len(coins)
    tree = [set() for _ in range(n)]
    leavesToBeRemoved = collections.deque()

    for u, v in edges:
      tree[u].add(v)
      tree[v].add(u)

    for u in range(n):
      # Remove the leaves that don't have coins.
      while len(tree[u]) == 1 and coins[u] == 0:
        v = tree[u].pop()
        tree[v].remove(u)
        u = v  # Walk up to its parent.
      # After trimming leaves without coins, leaves with coins may satisfy
      # `leavesToBeRemoved`.
      if len(tree[u]) == 1:  # coins[u] must be 1.
        leavesToBeRemoved.append(u)

    # Remove each remaining leaf node and its parent. The remaining nodes are
    # the ones that must be visited.
    for _ in range(2):
      for _ in range(len(leavesToBeRemoved)):
        u = leavesToBeRemoved.popleft()
        if tree[u]:
          v = tree[u].pop()
          tree[v].remove(u)
          if len(tree[v]) == 1:  # It's a leaf.
            leavesToBeRemoved.append(v)

    return sum(len(children) for children in tree)


# Link: https://leetcode.com/problems/maximum-spending-after-buying-items/description/
class Solution:
  def maxSpending(self, values: List[List[int]]) -> int:
    items = sorted(item for shop in values for item in shop)
    return sum(item * d for d, item in enumerate(items, 1))


# Link: https://leetcode.com/problems/painting-the-walls/description/
class Solution:
  def paintWalls(self, cost: List[int], time: List[int]) -> int:
    kMax = 500_000_000
    n = len(cost)
    # dp[i] := the minimum cost to paint i walls by the painters so far
    dp = [0] + [kMax] * n

    for c, t in zip(cost, time):
      for walls in range(n, 0, -1):
        dp[walls] = min(dp[walls], dp[max(walls - t - 1, 0)] + c)

    return dp[n]


# Link: https://leetcode.com/problems/painting-the-walls/description/
class Solution:
  def paintWalls(self, cost: List[int], time: List[int]) -> int:
    n = len(cost)

    @functools.lru_cache(None)
    def dp(i: int, walls: int) -> int:
      """Returns the minimum cost to paint j walls by painters[i..n)."""
      if walls <= 0:
        return 0
      if i == n:
        return math.inf
      pick = cost[i] + dp(i + 1, walls - time[i] - 1)
      skip = dp(i + 1, walls)
      return min(pick, skip)

    return dp(0, n)


# Link: https://leetcode.com/problems/amount-of-new-area-painted-each-day/description/
from sortedcontainers import SortedList


class Solution:
  def amountPainted(self, paint: List[List[int]]) -> List[int]:
    minDay = min(s for s, e in paint)
    maxDay = max(e for s, e in paint)
    ans = [0] * len(paint)
    # Stores the indices of paints that are available now.
    runningIndices = SortedList()
    events = []  # (day, index, type)

    for i, (start, end) in enumerate(paint):
      events.append((start, i, 1))  # 1 := entering
      events.append((end, i, -1))  # -1 := leaving

    events.sort()

    i = 0  # events' index
    for day in range(minDay, maxDay):
      while i < len(events) and events[i][0] == day:
        day, index, type = events[i]
        if type == 1:
          runningIndices.add(index)
        else:
          runningIndices.remove(index)
        i += 1
      if runningIndices:
        ans[runningIndices[0]] += 1

    return ans


# Link: https://leetcode.com/problems/find-the-k-sum-of-an-array/description/
class Solution:
  def kSum(self, nums: List[int], k: int) -> int:
    maxSum = sum(num for num in nums if num > 0)
    absNums = sorted(abs(num) for num in nums)
    # (the next maximum sum, the next index i)
    maxHeap = [(-(maxSum - absNums[0]), 0)]
    nextMaxSum = maxSum

    for _ in range(k - 1):
      nextMaxSum, i = heapq.heappop(maxHeap)
      nextMaxSum *= -1
      if i + 1 < len(absNums):
        heapq.heappush(maxHeap, (-(nextMaxSum - absNums[i + 1]), i + 1))
        heapq.heappush(
            maxHeap, (-(nextMaxSum - absNums[i + 1] + absNums[i]), i + 1))

    return nextMaxSum


# Link: https://leetcode.com/problems/maximum-number-of-groups-getting-fresh-donuts/description/
class Solution:
  def maxHappyGroups(self, batchSize: int, groups: List[int]) -> int:
    happy = 0
    freq = [0] * batchSize

    for g in groups:
      g %= batchSize
      if g == 0:
        happy += 1
      elif freq[batchSize - g]:
        freq[batchSize - g] -= 1
        happy += 1
      else:
        freq[g] += 1

    @functools.lru_cache(None)
    def dp(freq: int, remainder: int) -> int:
      """Returns the maximum number of partitions can be formed."""
      ans = 0
      if any(freq):
        for i, f in enumerate(freq):
          if f:
            ans = max(ans, dp(freq[:i] + (f - 1,) +
                              freq[i + 1:], (remainder + i) % batchSize))
        if remainder == 0:
          ans += 1
      return ans

    return happy + dp(tuple(freq), 0)


# Link: https://leetcode.com/problems/building-boxes/description/
class Solution:
  def minimumBoxes(self, n: int) -> int:
    nBoxes = 0
    nextTouchings = 0  # j
    currLevelBoxes = 0  # 1 + 2 + ... + j

    # Find the minimum j s.t. `nBoxes` = 1 + (1 + 2) + ... + (1 + 2 + ... + j)
    # >= n
    while nBoxes < n:
      nextTouchings += 1
      currLevelBoxes += nextTouchings
      nBoxes += currLevelBoxes

    # If nBoxes = n, the answer is `currLevelBoxes` = 1 + 2 + ... + j.
    if nBoxes == n:
      return currLevelBoxes

    # Otherwise, need to remove the boxes in the current level and rebuild it.
    nBoxes -= currLevelBoxes
    currLevelBoxes -= nextTouchings
    nextTouchings = 0

    while nBoxes < n:
      nextTouchings += 1
      nBoxes += nextTouchings

    return currLevelBoxes + nextTouchings


# Link: https://leetcode.com/problems/lexicographically-smallest-beautiful-string/description/
class Solution:
  def smallestBeautifulString(self, s: str, k: int) -> str:
    chars = list(s)

    for i in reversed(range(len(chars))):
      chars[i] = chr(ord(chars[i]) + 1)
      while self._containsPalindrome(chars, i):
        chars[i] = chr(ord(chars[i]) + 1)
      if chars[i] < chr(ord('a') + k):
        # If s[i] is among the first k letters, then change the letters after
        # s[i] to the smallest ones that don't form any palindrome substring.
        return self._changeSuffix(chars, i + 1)

    return ''

  def _containsPalindrome(self, chars: List[str], i: int) -> bool:
    """Returns True if chars[0..i] contains palindrome."""
    return (i > 0 and chars[i] == chars[i - 1]) or \
        (i > 1 and chars[i] == chars[i - 2])

  def _changeSuffix(self, chars: List[str], i: int) -> str:
    """
    Returns a string, where replacing sb[i..n) with the smallest possible
    letters don't form any palindrome substring.
    """
    for j in range(i, len(chars)):
      chars[j] = 'a'
      while self._containsPalindrome(chars, j):
        chars[j] = chr(ord(chars[j]) + 1)
    return ''.join(chars)


# Link: https://leetcode.com/problems/maximum-balanced-subsequence-sum/description/
class FenwickTree:
  def __init__(self, n: int):
    self.vals = [0] * (n + 1)

  def update(self, i: int, val: int) -> None:
    """Updates the maximum the sum of subsequence ending in (i - 1) with `val`."""
    while i < len(self.vals):
      self.vals[i] = max(self.vals[i], val)
      i += FenwickTree.lowbit(i)

  def get(self, i: int) -> int:
    """Returns the maximum the sum of subsequence ending in (i - 1)."""
    res = 0
    while i > 0:
      res = max(res, self.vals[i])
      i -= FenwickTree.lowbit(i)
    return res

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i


class Solution:
  def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
    # Let's define maxSum[i] := subsequence with the maximum sum ending in i
    # By observation:
    #    nums[i] - nums[j] >= i - j
    # => nums[i] - i >= nums[j] - j
    # So, if nums[i] - i >= nums[j] - j, where i > j,
    # maxSum[i] = max(maxSum[i], maxSum[j] + nums[i])
    ans = -math.inf
    tree = FenwickTree(len(nums))

    for _, i in sorted([(num - i, i) for i, num in enumerate(nums)]):
      subseqSum = tree.get(i) + nums[i]
      tree.update(i + 1, subseqSum)
      ans = max(ans, subseqSum)

    return ans


# Link: https://leetcode.com/problems/maximal-rectangle/description/
class Solution:
  def maximalRectangle(self, matrix: List[List[str]]) -> int:
    if not matrix:
      return 0

    ans = 0
    hist = [0] * len(matrix[0])

    def largestRectangleArea(heights: List[int]) -> int:
      ans = 0
      stack = []

      for i in range(len(heights) + 1):
        while stack and (i == len(heights) or heights[stack[-1]] > heights[i]):
          h = heights[stack.pop()]
          w = i - stack[-1] - 1 if stack else i
          ans = max(ans, h * w)
        stack.append(i)

      return ans

    for row in matrix:
      for i, num in enumerate(row):
        hist[i] = 0 if num == '0' else hist[i] + 1
      ans = max(ans, largestRectangleArea(hist))

    return ans


