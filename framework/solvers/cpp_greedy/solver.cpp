#include<bits/stdc++.h>
using namespace std;

vector<int> solve_mis_greedy(int n, const vector< pair<int,int> >& edges){

    int m = edges.size();
    
    vector<vector<int>> adj(n);

    for(auto& edge : edges){
        int u = edge.first;
        int v = edge.second;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> degree(n, 0);
    for(int i = 0; i < n; i++) degree[i] = adj[i].size();

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    for(int i = 0; i < n; i++) pq.push({degree[i], i});

    vector<int> ans;

    while (!pq.empty()){
        auto [deg, u] = pq.top();
        pq.pop();

        if (degree[u] != deg) continue;
        ans.push_back(u);
        degree[u] = -1;

        for (int v:adj[u]){
            degree[v] = -1;
            for (int w:adj[v]){
                if (degree[w] > 0) {
                    degree[w]--;
                    pq.push({degree[w], w});
                }
            }
        }
    }

    return ans;

}