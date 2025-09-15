# dp
## 数位dp
[P2602 [ZJOI2010] 数字计数 - 洛谷](https://www.luogu.com.cn/problem/P2602)
```cpp
/*
234 
第一次：计算出000~199的高位和低位贡献以及200~234的高位贡献
第二次：计算出200~229的高位和低位贡献以及230~234的高位贡献
第三次：计算出230~234
*/
void solve() {
    int l, r;
    cin >> l >> r;
    mi[0] = 1;
    for (int i = 1; i <= 13; ++i) {
        dp[i] = dp[i - 1] * 10 + mi[i - 1];
        mi[i] = 10 * mi[i - 1];
    }
    auto cul = [](int n, vector<int> &ans) {
        int tmp = n;
        vector<int> a(15);
        int len = 0;
        while (n) a[++len] = n % 10, n /= 10;
        for (int i = len; i >= 1; --i) {
            for (int j = 0; j < 10; ++j) ans[j] += dp[i - 1] * a[i];//i位之后的贡献
            for (int j = 0; j < a[i]; ++j) ans[j] += mi[i - 1];//i位的贡献
            tmp -= mi[i - 1] * a[i], ans[a[i]] += tmp + 1;//i位最大值特判
            ans[0] -= mi[i - 1];//前导0
        }
    };
    cul(r, ans1), cul(l - 1, ans2);
    for (int i = 0; i < 10; ++i)
        cout << ans1[i] - ans2[i] << ' ';
}
```

# 图论

## 树上问题
### 树的最大独立集
```cpp
void solve() {  
    int n;  
    cin >> n;  
    vector<vector<int> > ed(n + 1, vector<int>());  
    for (int i = 1; i < n; ++i) {  
        int u, v;  
        cin >> u >> v;  
        ed[u].push_back(v);  
        ed[v].push_back(u);  
    }  
    vector<vector<int> > dp(n + 1, vector<int>(2));  
    function<void(int,int)> dfs = [&](int u, int p) {  
        dp[u][0] = 0;  
        dp[u][1] = 1;  
        for (int v: ed[u]) {  
            if (v == p) continue;  
            dfs(v, u);  
            dp[u][0] += max(dp[v][0], dp[v][1]);  
            dp[u][1] += dp[v][0];  
        }  
    };  
    dfs(1, 0);  
}
```
### 树的直径
dfs无法求含有负权边的树
树形dp可以
#### 两次dfs
`如果需要求出一条直径上所有的节点，则可以在第二次 DFS 的过程中，记录每个点的前序节点，即可从直径的一端一路向前，遍历直径上所有的节点。`
```cpp
void solve() {  
    int n;  
    cin >> n;  
    vector<vector<int> > ed(n + 1, vector<int>());  
    for (int i = 1; i < n; ++i) {  
        int u, v;  
        cin >> u >> v;  
        ed[u].push_back(v), ed[v].push_back(u);  
    }  
    vector<int> dis(n + 1, 0);  
    int c = 0;  
    function<void(int,int)> dfs = [&](int u, int fa) {  //fa避免子节点回到父节点
        for (int v: ed[u]) {  
            if (v == fa) continue;  
            dis[v] = dis[u] + 1;  
            if (dis[v] > dis[c]) c = v;  
            dfs(v, u);  
        }  
    };  
    dfs(1, 0);  
    dis[c] = 0, dfs(c, 0);  
    cout << dis[c] << endl;  
}
```

#### 树形dp
`dp[u]代表从当前节点出发的最长路径`
`转移方程:dp[u] = max(dp[u], dp[v] + w(u, v))`
`由于只需要求最长直径，所以可以在更新dp[u]之前，更新d = max(d, dp[u] + dp[v] + w(u, v)`
```cpp
void solve() {  
    int n;  
    cin >> n;  
    vector<vector<int> > ed(n + 1, vector<int>());  
    for (int i = 1; i < n; ++i) {  
        int u, v;  
        cin >> u >> v;  
        ed[u].push_back(v), ed[v].push_back(u);  
    }  
    int d = 0;  
    vector<int> dp(n + 1);  
    function<void(int,int)> dfs = [&](int u,int fa) {  
        for (int v: ed[u]) {  
            if (v == fa) continue;  
            dfs(v, u);  
            d = max(d, dp[u] + dp[v] + 1);  
            dp[u] = max(dp[u], dp[v] + 1);  
        }  
    };  
    dfs(1, 0);  
    cout << d << endl;  
}
```


`d1代表这个节点最长的子树深度，d2代表这个节点次长的子树深度`
```cpp
void solve() {  
    int n;  
    cin >> n;  
    vector<vector<int> > ed(n + 1, vector<int>());  
    for (int i = 1; i < n; ++i) {  
        int u, v;  
        cin >> u >> v;  
        ed[u].push_back(v), ed[v].push_back(u);  
    }  
    int d = 0;  
    vector<int> d1(n + 1), d2(n + 1);  
    function<void(int,int)> dfs = [&](int u, int fa) {  
        d1[u] = d2[u] = 0;  
        for (int v: ed[u]) {  
            if (v == fa) continue;  
            dfs(v, u);  
            int t = d1[v] + 1;  
            if (t > d1[u])  
                d2[u] = d1[u], d1[u] = t;  
            else if (t > d2[u])  
                d2[u] = t;  
        }  
        d = max(d, d1[u] + d2[u]);  
    };  
    dfs(1 , 0);  
    cout << d << endl;  
}
```

#### 性质
若树上所有边边权为正，则树的所有直径的中点重合

### 例题
[Problem - 2107D - Codeforces](https://codeforces.com/problemset/problem/2107/D)
[Problem - 911F - Codeforces](https://codeforces.com/problemset/problem/911/F)
#### 一些trick
[Problem - 1336A - Codeforces](https://codeforces.com/problemset/problem/1336/A)
[Problem - 2109D - Codeforces](https://codeforces.com/problemset/problem/2109/D)

## 最短路
### Floyd
适用于任何图，不管有向无向，边权正负，但是最短路必须存在。（不能有个负环）
O(N^3)
```cpp
void solve() {  
    int n, m;  
    cin >> n >> m;  
    vector dis(n + 1, vector<int>(n + 1, 2e9));  
    for (int i = 1; i <= n; ++i) dis[i][i] = 0;  
    for (int i = 1; i <= m; ++i) {  
        int u, v, w;  
        cin >> u >> v >> w;  
        dis[u][v] = dis[v][u] = min(dis[u][v], w);  
    }  
    for (int k = 1; k <= n; ++k) {  
        for (int i = 1; i <= n; ++i) {  
            for (int j = 1; j <= n; ++j) {  
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j]);  
            }  
        }  
    }
    for (int i = 1; i <= n; i++) {
	    if (d[i][i] < 0) {
	        cout << "存在负环\n";
	    }
    }
}
```


### bellman-ford
每次对全部边进行一次松弛操作：`dis[v] = min(dis[v], dis[u] + w[u, v]`
因为任意两点之间的最短路长度不会超过n-1
所以只需要松弛n-1次
如果在第n次循坏还有松弛操作，则证明有负环存在
O(n * m)
```cpp
struct Edge {  
    int u, v, w;  
};  
  
vector<Edge> edge;  
  
int dis[MAXN], u, v, w;  
constexpr int INF = 0x3f3f3f3f;  
  
bool bellmanford(int n, int s) {  
    memset(dis, 0x3f, (n + 1) * sizeof(int));  
    dis[s] = 0;  
    bool flag = false; // 判断一轮循环过程中是否发生松弛操作  
    for (int i = 1; i <= n; i++) {  
        flag = false;  
        for (int j = 0; j < edge.size(); j++) {  
            u = edge[j].u, v = edge[j].v, w = edge[j].w;  
            if (dis[u] == INF) continue;  
            // 无穷大与常数加减仍然为无穷大  
            // 因此最短路长度为 INF 的点引出的边不可能发生松弛操作  
            if (dis[v] > dis[u] + w) {  
                dis[v] = dis[u] + w;  
                flag = true;  
            }  
        }  
        // 没有可以松弛的边时就停止算法  
        if (!flag) {  
            break;  
        }  
    }  
    // 第 n 轮循环仍然可以松弛时说明 s 点可以抵达一个负环  
    return flag;  
}
```

#### spfa
每一次循环其实真正能够进行松弛操作的边只有上一次松弛的点
没有负权边的时候用Dijkstra，有负权边且题目中的图没有特殊性质时，可以用spfa
```cpp
void solve() {  
    const int INF = 4e18;  
    int n, m;  
    cin >> n >> m;  
    vector ed(n + 1, vector<PII>());  
    for (int i = 1; i <= m; ++i) {  
        int u, v, w;  
        cin >> u >> v >> w;  
        ed[u].push_back({v, w});  
    }  
    vector<int> dis(n + 1, INF);  
    vector<bool> vis(n + 1, false);  
    vector<int> cnt(n + 1, 0);   //记录最短路经过的边数
    int f = 1;  
    auto spfa = [&]() {  
        dis[1] = 0;  
        vis[1] = true;
        queue<int> q;  
        q.push(1);  
        while (!q.empty()) {  
            int u = q.front();  
            q.pop(), vis[u] = false;  
            for (auto &[v, w]: ed[u]) {  
                if (dis[v] > dis[u] + w) {  
                    dis[v] = dis[u] + w;  
                    cnt[v] = cnt[u] + 1;  
                    if (cnt[v] >= n) {  
                        f = 0;  
                        return;  //到这个点的最短路距离已经>=n 有负环
                    }  
                    if (!vis[v]) q.push(v), vis[v] = true;  
                }  
            }  
        }  
    };  
    spfa();  
    if (!f || dis[n] == INF) {  
        cout << "impossible";  
    } else {  
        cout << dis[n];  
    }  
}
```
### Dijkstra
非负权图单源最短路
朴素：O(n^2)
```cpp
struct edge {  
    int v, w;  
};  
  
vector<edge> e[MAXN];  
int dis[MAXN], vis[MAXN];  
  
void dijkstra(int n, int s) {  
    memset(dis, 0x3f, (n + 1) * sizeof(int));  
    dis[s] = 0;  
    for (int i = 1; i <= n; i++) {  
        int u = 0, mind = 0x3f3f3f3f;  
        for (int j = 1; j <= n; j++)  //找到一个未确定最短路最短长度的点
            if (!vis[j] && dis[j] < mind) u = j, mind = dis[j];  
        vis[u] = true;  
        for (auto ed: e[u]) {  //将这个点所连的边全部松弛操作
            int v = ed.v, w = ed.w;  
            if (dis[v] > dis[u] + w) dis[v] = dis[u] + w;  
        }  
    }  
}
```

可以通过堆优化，每成功松弛一条边(u, v)就将v插入队中，然后直接取堆顶结点
O((n + m) logn)
```cpp
auto dijkstra = [&](int s = 1) {  
    auto cmp = [](const PII &a, const PII &b) {  
        return a.first > b.first;  
    };  
    priority_queue<PII, vector<PII>, decltype(cmp)> q(cmp);  
  
    dis[s] = 0;  
    q.push({0, s});  
    while (!q.empty()) {  
        int u = q.top().second;  
        q.pop();  
        if (vis[u]) continue;  
        vis[u] = 1;  
        for (auto ed: g[u]) {  
            int v = ed.first, w = ed.second;  
            if (dis[v] > dis[u] + w) {  
                pre[v] = u;  
                dis[v] = dis[u] + w;  
                q.push({dis[v], v});  
            }  
        }  
    }  
};
```
# 数据结构

## 线段树
[Problem - 1234D - Codeforces](https://codeforces.com/problemset/problem/1234/D)
### 建树
数组大小为N，数组树开4N
```cpp
vector<int> a(N, 0), d(4 * N, 0);

void build(int s, int e, int p) { 
	// 对 [s,e] 区间建立线段树,当前根的编号为 p
    if (s == e) {  
        d[p] = a[s];  
        return;  
    }  
    int m = s + ((e - s) >> 1);
    build(s, m, p * 2), build(m + 1, e, p * 2 + 1);  
    // 递归对左右区间建树
    d[p] = d[p * 2] + d[p * 2 + 1];  
}
```

### 区间查询
```cpp
int getsum(int l, int r,int s, int e, int p) {  
	// [l, r] 为查询区间, [s, e] 为当前节点包含的区间, p 为当前节点的编号
    if (l <= s && e <= r) {
    // 当前区间为询问区间的子集时直接返回当前区间的和  
        return d[p];  
    }  
    int m = s + ((e - s) >> 1), sum = 0;  
    if (l <= m) sum += getsum(l, r, s, m, p * 2);  
    if (r > m) sum += getsum(l, r, m + 1, e, p * 2 + 1);  
    return sum;  
}
```

### 区间修改
```cpp
void update(int l, int r, int c, int s, int t, int p) {  
    if (l <= s && t <= r) {  
        d[p] += (t - s + 1) * c;  
        b[p] += c;  
        return;  
    }  
    int m = s + ((t - s) >> 1);  
    if (b[p] && s != t) {  
        d[p * 2] += b[p] * (m - s + 1), d[p * 2 + 1] += b[p] * (t - m);  
        b[p * 2] += b[p], b[p * 2 + 1] += b[p];  
        b[p] = 0;  
    }  
    if (l <= m) update(l, r, c, s, m, p * 2);  
    if (r > m) update(l, r, c, m + 1, t, p * 2 + 1);  
    d[p] = d[p * 2] + d[p * 2 + 1];  
}
```

### 区间查询（求和）
```cpp
int getsum(int l,int r,int s, int t,int p) {  
    if (l <= s && t <= r) return d[p];  
    int m = s + ((t - s) >> 1);  
    if (b[p]) {  
        d[p * 2] += b[p] * (m - s + 1), d[p * 2 + 1] += b[p] * (t - m);  
        b[p * 2] += b[p], b[p * 2 + 1] += b[p];  
        b[p] = 0;  
    }  
    int sum = 0;  
    if (l <= m) sum += getsum(l, r, s, m, p * 2);  
    if (r > m) sum += getsum(l, r, m + 1, t, p * 2 + 1);  
    return sum;  
}
```

### 模板
```cpp
template<typename T>  
class SegTreeLazyRangeAdd {  
    vector<T> tree, lazy;  
    vector<T> *arr;  
    int n, root, n4, end;  
  
    void maintain(int cl, int cr, int p) {  
        int cm = cl + (cr - cl) / 2;  
        if (cl != cr && lazy[p]) {  
            lazy[p * 2] += lazy[p];  
            lazy[p * 2 + 1] += lazy[p];  
            tree[p * 2] += lazy[p] * (cm - cl + 1);  
            tree[p * 2 + 1] += lazy[p] * (cr - cm);  
            lazy[p] = 0;  
        }  
    }  
  
    T range_sum(int l, int r, int cl, int cr, int p) {  
        if (l <= cl && cr <= r) return tree[p];  
        int m = cl + (cr - cl) / 2;  
        T sum = 0;  
        maintain(cl, cr, p);  
        if (l <= m) sum += range_sum(l, r, cl, m, p * 2);  
        if (r > m) sum += range_sum(l, r, m + 1, cr, p * 2 + 1);  
        return sum;  
    }  
  
    void range_add(int l, int r, T val, int cl, int cr, int p) {  
        if (l <= cl && cr <= r) {  
            lazy[p] += val;  
            tree[p] += (cr - cl + 1) * val;  
            return;  
        }  
        int m = cl + (cr - cl) / 2;  
        maintain(cl, cr, p);  
        if (l <= m) range_add(l, r, val, cl, m, p * 2);  
        if (r > m) range_add(l, r, val, m + 1, cr, p * 2 + 1);  
        tree[p] = tree[p * 2] + tree[p * 2 + 1];  
    }  
  
    void build(int s, int t, int p) {  
        if (s == t) {  
            tree[p] = (*arr)[s];  
            return;  
        }  
        int m = s + (t - s) / 2;  
        build(s, m, p * 2);  
        build(m + 1, t, p * 2 + 1);  
        tree[p] = tree[p * 2] + tree[p * 2 + 1];  
    }  
  
public:  
    explicit SegTreeLazyRangeAdd<T>(vector<T> v) {  
        n = v.size();  
        n4 = n * 4;  
        tree = vector<T>(n4, 0);  
        lazy = vector<T>(n4, 0);  
        arr = &v;  
        end = n - 1;  
        root = 1;  
        build(0, end, 1);  
        arr = nullptr;  
    }  
  
    void show(int p, int depth = 0) {  
        if (p > n4 || tree[p] == 0) return;  
        show(p * 2, depth + 1);  
        for (int i = 0; i < depth; ++i) cout << '\t';  
        cout << tree[p] << ':' << lazy[p] << endl;  
        show(p * 2 + 1, depth + 1);  
    }  
  
    T range_sum(int l, int r) { return range_sum(l, r, 0, end, root); }  
  
    void range_add(int l, int r, T val) { range_add(l, r, val, 0, end, root); }  
};
```

## ST表
```cpp
template<typename T>  
class SparseTable {  
    using VT = vector<T>;  
    using VVT = vector<VT>;  
    using func_type = function<T(const T &, const T &)>;  
  
    VVT ST;  
  
    static T default_func(const T &t1, const T &t2) { return max(t1, t2); }  
  
    func_type op;  
  
public:  
    SparseTable(const vector<T> &v, func_type _func = default_func) {  
        op = _func;  
        int len = v.size(), l1 = ceil(log2(len)) + 1;  
        ST.assign(len, VT(l1, 0));  
        for (int i = 0; i < len; ++i) {  
            ST[i][0] = v[i];  
        }  
        for (int j = 1; j < l1; ++j) {  
            int pj = (1 << (j - 1));  
            for (int i = 0; i + pj < len; ++i) {  
                ST[i][j] = op(ST[i][j - 1], ST[i + (1 << (j - 1))][j - 1]);  
            }  
        }  
    }  
  
    T query(int l, int r) {  
        int lt = r - l + 1;  
        int q = floor(log2(lt));  
        return op(ST[l][q], ST[r - (1 << q) + 1][q]);  
    }  
};
```

# 背包问题
## 01背包
```cpp
void solve() {  
    int n, m;  
    cin >> n >> m;  
    vector<int> c(n + 1, 0), v(n + 1, 0);  
    vector<vector<int> > dp(n + 1, vector<int>(m + 1, 0));  
    for (int i = 1; i <= n; ++i) {  
        cin >> c[i] >> v[i];  
    }  
    for (int i = 1; i <= n; ++i) {  
        for (int j = 1; j <= m; ++j) {  
            dp[i][j] = dp[i - 1][j];  
            if (j >= c[i])  
                dp[i][j] = max(dp[i][j], dp[i - 1][j - c[i]] + v[i]);  
        }  
    }  
    cout << dp[n][m];  
}
```
### 多个不同的数字组合成某个特定的数的方案数
```cpp
	vector<int> dp(sum + 1);  
	dp[0] = 1;  
	for (int i = 0; i < c.size(); ++i) {  
	    for (int j = sum; j >= 0; --j) {  
	        if (j + c[i] <= sum)  
	            dp[j + c[i]] = (dp[j + c[i]] + dp[j]) % MOD;  
	    }  
	}
```

## 完全背包
```cpp
void solve() {  
    int n, m;  
    cin >> n >> m;  
    vector<int> c(n + 1, 0), v(n + 1, 0);  
    vector<vector<int> > dp(n + 1, vector<int>(m + 1, 0));  
    for (int i = 1; i <= n; ++i) {  
        cin >> c[i] >> v[i];  
    }  
    for (int i = 1; i <= n; ++i) {  
        for (int j = 1; j <= m; ++j) {  
            dp[i][j] = dp[i - 1][j];  
            if (j >= c[i]) {  
                dp[i][j] = max(dp[i][j], dp[i - 1][j - c[i]] + v[i]);  
                dp[i][j] = max(dp[i][j], dp[i][j - c[i]] + v[i]);  
            }  
        }  
    }  
    cout << dp[n][m];  
}
```
## 多重背包
```cpp
void solve() {
    int n, m;
    cin >> n >> m;
    vector<int> c(1, 0), v(1, 0);
    for (int i = 1; i <= n; ++i) {
        int a, b, s;
        cin >> a >> b >> s;
        int k = 1;
        while (k <= s) {
            c.push_back(a * k);
            v.push_back(b * k);
            s -= k;
            k *= 2;
        }
        if (s > 0) {
            c.push_back(a * s);
            v.push_back(b * s);
        }
    }
    n = c.size() - 1;
    vector<int> dp(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        for (int j = m; j >= c[i]; --j) {
            dp[j] = max(dp[j], dp[j - c[i]] + v[i]);
        }
    }
    cout << dp[m];
}
```
## 分组背包
```cpp
void solve() {  
    int n, m;  
    cin >> n >> m;  
    vector<int> s(n + 1, 0);  
    vector<vector<int> > c(n + 1, vector<int>(1, 0)), v(n + 1, vector<int>(1, 0));  
    for (int i = 1; i <= n; ++i) {  
        cin >> s[i];  
        for (int j = 1; j <= s[i]; ++j) {  
            int a, b;  
            cin >> a >> b;  
            c[i].push_back(a);  
            v[i].push_back(b);  
        }  
    }  
    vector<int> dp(m + 1);  
    for (int i = 1; i <= n; ++i) {  
        for (int j = m; j >= 0; --j) {  
            for (int k = 1; k <= s[i]; ++k) {  
                if (j - c[i][k] >= 0) dp[j] = max(dp[j], dp[j - c[i][k]] + v[i][k]);  
            }  
        }  
    }  
    cout << dp[m];  
}
```

# 数论
## 线性筛
```cpp
vector<int> pri;  
bool not_prime[N];  
vector<int> spf(N);  
  
void pre(int n) {  
    for (int i = 2; i <= n; ++i) {  
        if (!not_prime[i]) {  
            pri.push_back(i);  
            spf[i] = i;  
        }  
        for (int pri_j: pri) {  
            if (i * pri_j > n) break;  
            not_prime[i * pri_j] = true;  
            spf[i * pri_j] = pri_j;  
            if (i % pri_j == 0) {  
                break;  
            }  
        }  
    }  
}
```

## 分解质因数
```cpp
vector<PII> breakdown(int N) {  
    vector<PII> result;  
    while (N > 1) {  
        int div = spf[N], cnt = 0;  
        while (N % div == 0) N /= div, ++cnt;  
        result.push_back({div, cnt});  
    }  
    return result;  
}
```

## 快速幂
```cpp
function<int (int, int)> quick_pow = [](int x,int p) {  
    int res = 1;  
    while (p) {  
        if (p & 1) {  
            res = res * x % MOD;  
        }  
        p >>= 1;  
        x = x * x % MOD;  
    }  
    return res;  
};
```
# 杂项


## 单调队列
```cpp
void solve() {  
    int n, k;  
    cin >> n >> k;  
    vector<int> a(n);  
    for (int i = 0; i < n; ++i) cin >> a[i];  
    deque<int> dq;  
    vector<int> result;  
    for (int i = 0; i < n; ++i) {  
        if (!dq.empty() && dq.front() < i - k + 1) {  
            dq.pop_front();  
        }  
  
        while (!dq.empty() && a[dq.back()] > a[i]) {  
            dq.pop_back();  
        }  
  
        dq.push_back(i);  
  
        if (i >= k - 1) {  
            result.push_back(a[dq.front()]);  
        }  
    }  
}
```