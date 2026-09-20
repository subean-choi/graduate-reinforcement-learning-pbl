# 강화학습 PBL

---
- [Toy model] 추가 제약 정리
- Dual Sourcing + Lead time 불확실성
- 강화학습 결과 정리
- 모델 정리
---
---
- [01] Course introduction_organized
- [02] Linear programming
- [03] Mixed Integer Programming
- RLPBL_solver
- RLPBL_Simopt
- **[06] Markov Decision Processes**
- [07] Temporal Difference Learning
- [08] Deep Q-Network
- REINFORCE
- Actor-Critic

---

# Toy model] 추가 제약 정리

---
## 목차
---
[image omitted: temporary Notion asset]
## **제약 1. 계절성 수요를 고려한 재고관리**

### 1. 연구 주제 소개
- 기본 toy model은 수요 분포가 모든 기간 동안 동일하다고 가정
- 하지만 현실에서는 시간에 따라 수요가 변함
- 본 주제에서는 **계절성 수요**를 추가로 고려
- 즉, 시점에 따라 평균 수요가 달라지는 재고관리 문제를 다룸

---
#### 1-1. 추가적으로 고려한 제약
- 기존 수요 가정
$$
d_{t} \thicksim Poisson(25)
$$
- 확장 수요 가정
$$
d_{t} \thicksim Poisson(\lambda_{t})
$$
- λt는 시간 t에 따라 변하는 평균 수요
- 성수기에는 λt 증가
- 비수기에는 λt 감소
#### 1-2. 왜 고려했는가
- 현실의 수요는 항상 일정하지 않음
- 계절, 이벤트, 특정 기간에 따라 수요가 달라짐
- 고정된 주문 정책은 수요 변화에 유연하게 대응하기 어려움
- 강화학습은 현재 재고뿐만 아니라 현재 시점까지 고려하여 주문 정책을 학습할 수 있음
#### 1-3. 어떤 현실 문제를 모형화했는가
- 여름철 음료, 아이스크림 수요 증가
- 겨울철 난방용품 수요 증가
- 명절 전 식품, 선물세트 수요 증가
- 학기 초 문구류 수요 증가
- 즉, **시간에 따라 수요가 달라지는 상품 재고관리 문제**를 모형화
---
### 2. MDP 환경 수학적 정의
---
#### State
$$
s_{t} = (x_{t},t)
$$
- xt : 현재 재고량
- t : 현재 시점
- 기존 모형은 재고량만 고려했지만, 계절성 수요에서는 시간이 중요
- 같은 재고량이라도 성수기인지 비수기인지에 따라 주문량이 달라질 수 있음
---
#### Action → 현재 시점에 몇 개를 주문할지 결정
$$
a_{t}=q_{t}\\
q_t \in \{0,1,\dots,M\}
$$
- qt: 시점 t의 주문량
- 한 번에 최대 M개까지 주문 가능
---
#### Demand
$$
d_{t} \thicksim Poisson(\lambda_{t})
$$
- λt: 시점에 따라 변하는 평균 수요
- 계절성 수요에서는 평균 수요가 항상 같지 않음
- 예시 :
$$
\lambda_t =
\begin{cases}
\lambda_{high}, & \text{성수기} \\
\lambda_{low}, & \text{비수기}
\end{cases}
$$
---
#### Transition
$$
x_{t+1} = \min \left(C, \max(x_t + q_t - d_t, 0)\right)\\z_t = \max(d_t - (x_t + q_t), 0)
$$
- xt+1: 다음 재고량
- zt: 결품량
- C: 창고 물량
- 재고는 0보다 작아질 수 없고, 창고 용량 C를 넘을 수 없음
---
#### Reward
$$
r_t = - \left(Ky_t + cq_t + hx_{t+1} + pz_t\right)
$$
- Ky_t: 주문 고정비
- cq_t: 단위 주문 비용
- hx_(t+1): 재고 유지 비용
- pz_t: 결품 비용
- 재고관리 문제는 비용을 최소화하는 문제
- 강화학습은 보상을 최대화하는 구조이므로, 비용의 음수를 reward로 사용
---
#### Objective
$$
\max_{\pi} E \left[\sum_{t=0}^{T-1} r_t \right]
$$
- 강화학습의 목표는 전체 기간 동안 받을 보상의 합을 최대화하는 것
- 하지만 reward가 비용의 음수이므로, 실제 의미는 총비용 최소화
즉,
$$
\min_{\pi} E \left[
\sum_{t=0}^{T-1}
\left(Ky_t + cq_t + hx_{t+1} + pz_t\right)
\right]
$$
---
## 제약 2. 입고 지연을 고려한 재고관리

### 1. 연구 주제 소개
- 기본 toy model은 주문한 물량이 즉시 입고된다고 가정
- 하지만 현실에서는 주문 후 생산, 배송, 검수 등의 시간이 필요함
- 본 주제에서는 **입고 지연 lead time**을 추가로 고려
- 즉, 주문한 물량이 일정 기간 뒤에 입고되는 재고관리 문제를 다룸

---
#### 1-1. 추가적으로 고려한 제약
- 기존 모형
- 주문하면 당일 바로 입고
- 확장 모형
- 주문 후 L기간 뒤에 입고
- 시점 t에 주문한 qt는 t+L 시점에 입고
$$
q_t \rightarrow t+L \text{ 시점에 입고}
$$
- 아직 도착하지 않은 주문량을 pipeline inventory로 관리
#### 1-2. 왜 고려했는가
- 현실에서는 주문 즉시 상품이 도착하지 않음
- 제조, 포장, 운송, 검수 과정에서 시간이 걸림
- 입고 지연이 있으면 수요가 발생한 뒤 주문해도 늦을 수 있음
- 따라서 미래 수요를 예측해 미리 주문하는 정책이 필요함
#### 1-3. 어떤 현실 문제를 모형화했는가
- 제조업 부품 조달
- 해외 수입 상품 재고관리
- 온라인 쇼핑몰 물류센터
- 병원 의약품 및 의료소모품 재고관리
- 자동차 부품 공급망 관리
- 즉, **주문 후 일정 시간이 지나야 재고로 사용할 수 있는 공급망 문제**를 모형화
---
### 2. MDP 환경 수학적 정의
---
#### State
lead time이 L일 때 상태는 다음과 같이 정의
$$
s_t = \left(x_t, t, u_t^{(1)}, u_t^{(2)}, \dots, u_t^{(L)}\right)
$$
- xt: 현재 재고량
- t: 현재 시점
- ut(1): 다음 기간에 입고될 주문량
- ut(L): L기간 후 입고될 주문량
- 상태는 **현재 재고량**, **현재 시점**, **아직 도착하지 않은 주문량**으로 정의
- 입고 지연이 있으면 주문한 물량이 바로 사용되지 않음
- 따라서 앞으로 도착할 주문량까지 state에 포함해야 함
---
#### Action
$$
a_t = q_t\\q_t \in \{0, 1, 2, \dots, M\}
$$
- qt: 현재 시점의 주문량
- 단, 주문한 물량은 즉시 재고로 사용되지 않음
- L기간 뒤 입고
---
#### Demand
$$
d_t \sim P(D)\\
$$
예시 1 :
$$
d_t \sim \text Uniform(0, 50)
$$
예시 2:
$$
d_t \sim Poisson(25)
$$
---
#### Transition
- 먼저 도착 예정이었던 주문량이 입고됨
- 이번 시점에 도착하는 물량은 ut(1)
- 즉, 과거에 주문했던 물량 중 도착 시점이 된 물량이 현재 재고에 추가됨
$$
arrival_t = u_t^{(1)}
$$
- 입고 후 재고
- 창고 용량 C를 넘을 수 없기 때문에 min(C,⋅) 사용
$$
\tilde{x}_t = \min(C, x_t + arrival_t)
$$
- 수요 발생 후 다음 재고
- 입고 후 재고에서 수요만큼 빠짐
- 재고는 0보다 작아질 수 없음
$$
x_{t+1} = \max(\tilde{x}_t - d_t, 0)
$$
- 결품량
- 수요가 입고 후 재고보다 크면 부족한 만큼 결품 발생
- 수요를 모두 만족하면 결품은 0
$$
z_t = \max(d_t - \tilde{x}_t, 0)
$$
- 새 주문량 qt는 파이프라인 마지막에 들어감
$$
u_{t+1}^{(L)} = q_t
$$
- 기존 파이프라인은 한 칸씩 앞으로 이동
$$
u_{t+1}^{(i)} = u_t^{(i+1)}, \quad i = 1, 2, \dots, L-1
$$
---
#### Reward
$$
r_t = - \left(Ky_t + cq_t + hx_{t+1} + pz_t\right)
$$
- Ky_t: 주문 고정비
- cq_t: 단위 주문 비용
- hx_(t+1): 재고 유지 비용
- pz_t: 결품 비용
---
#### Objective
$$
\max_{\pi} E \left[\sum_{t=0}^{T-1} r_t \right]
$$
즉,
$$
\min_{\pi} E \left[
\sum_{t=0}^{T-1}
\left(Ky_t + cq_t + hx_{t+1} + pz_t\right)
\right]
$$
---

---

# Dual Sourcing + Lead time 불확실성

---
## 목차
---
## 논문 1. Dual Sourcing with Stochastic Lead Times

---
### 1. 논문에서 다루는 재고관리 문제
- 단일 제품 재고관리 문제
- 하나의 공급처가 아니라 **두 개의 공급처**를 고려
[image omitted: temporary Notion asset]
- 공급 구조 :
<table header-row="true">

<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>Server 1</td>
<td>일반 공급처의 첫 번째 처리 단계</td>
</tr>
<tr>
<td>Server 2</td>
<td>최종 입고 전 두 번째 처리 단계</td>
</tr>
<tr>
<td>Stock</td>
<td>완제품 재고 창고</td>
</tr>
<tr>
<td>Demand</td>
<td>고객 수요</td>
</tr>
</table>
- 일반 공급처 normal source
- 비용이 낮지만 리드타임이 김
- 일반 공급처 주문 :  Server 1 → Server 2 → Stock
- 긴급 공급처 emergency source
- 비용이 높지만 더 빠르게 입고 가능
- 긴급 공급처 주문 :  Server 2 → Stock
- 수요는 Poisson demand
- 재고 부족은 lost sales가 아니라 **backlogging**으로 처리
- 리드타임은 고정값이 아니라 queueing system에 의해 확률적으로 결정
### 2. 기존 toy model과의 차이
<table header-row="true">

<tr>
<td>**구분**</td>
<td>**기존 toy model**</td>
<td>**논문 모델**</td>
</tr>
<tr>
<td>공급처</td>
<td>1개</td>
<td>2개</td>
</tr>
<tr>
<td>주문 결정</td>
<td>주문량 (q_t)</td>
<td>일반 주문 / 긴급 주문 선택</td>
</tr>
<tr>
<td>리드타임</td>
<td>없음 또는 고정</td>
<td>확률적 리드타임</td>
</tr>
<tr>
<td>입고 구조</td>
<td>단순 입고</td>
<td>queue를 거쳐 입고</td>
</tr>
<tr>
<td>재고 부족</td>
<td>lost sales, 결품 변수 (z_t)</td>
<td>backlog</td>
</tr>
<tr>
<td>상태</td>
<td>현재 재고 (x_t)</td>
<td>순재고 + stage별 처리 중 주문</td>
</tr>
<tr>
<td>비용</td>
<td>주문비, 보유비, 결품비</td>
<td>공급처별 주문비, stage별 보유비, 완제품 보유비, backlog 비용</td>
</tr>
<tr>
<td>정책 비교</td>
<td>휴리스틱, RL</td>
<td>optimal policy, heuristic policy</td>
</tr>
</table>
### 3. MDP 환경 정의
- **state**
>
$$
s_t = (IN(t), N_2(t), N_1(t))
$$
- IN(t) : 순재고
- 현재 보유 재고에서 backlog를 뺀 값
- 지금 재고가 얼마나 있는지
- N1(t) : server 1에서 대기 또는 처리 중인 주문 수
- 일반 공급처 쪽 첫 번째 단계에 주문이 얼마나 밀려 있는지
- N2(t) : server 2에서 대기 또는 처리 중인 주문 수
- 두 번째 단계에 주문이 얼마나 밀려 있는지
- **Action**
>
논문에서의 의사결정은 단순히 주문량 하나를 고르는 것 X
- 일반 공급처에 주문할지
- 긴급 공급처에 주문할지
- 얼마나 주문할지
- 아무것도 하지 않을지를 결정
$$
a_t = (q_t^N, q_t^E)
$$
- qt\^N : normal source 주문량
- qt\^E : emergency source 주문량
- **Demand **
>
논문에서는 수요가 Poisson process
$$
d_t \sim Poisson(\lambda)
$$
- λ : 평균 수요율
- 수요는 확률적으로 발생
- 수요가 재고보다 많으면 backlog가 증가
- **Lead Time **
>
🌟 논문의 핵심은 리드타임이 고정되어 있지 않다는 점
→ 그때그떄 걸리는 시간이 랜덤하게 달라짐
- 일반 공급처:
- server 1 → server 2 → 재고
- 긴급 공급처:
- server 1을 건너뛰고 server 2로 바로 이동
→   처리시간은 exponential distribution을 따름
$$
T_i \sim Exp(\mu_i), \quad i = 1,2
$$
- μi : server i의 처리율
- 처리시간이 확률적이므로 실제 리드타임도 확률적으로 변함
- **Transition**
>
논문에서는 사건이 발생할 때마다 상태가 바뀜
- 주요 사건  세 가지 :
1. 수요 발생
2. server 1 처리 완료
3. server 2 처리 완료
1. 수요 발생 → 수요가 발생하면 순재고가 감소
$$
IN(t+1) = IN(t) - d_t
$$
- 재고가 충분하면 재고 감소
- 재고가 부족하면 backlog 증가
2. server 1 처리 완료 → server 1에서 처리가 끝나면 주문이 server 2로 이동
$$
N_1(t+1) = N_1(t) - 1
$$
$$
N_2(t+1) = N_2(t) + 1
$$
3. server 2 처리 완료 → server 2에서 처리가 끝나면 제품이 재고로 입고
$$
N_2(t+1) = N_2(t) - 1
$$
$$
IN(t+1) = IN(t) + 1
$$
- **Cost**
>
논문의 비용은 다음 요소로 구성 :
- c1 : normal source 주문 비용
- c2 : emergency source 주문 비용
- c2\>c1
- h1 : server 1에 있는 주문의 holding cost
- h2 : server 2에 있는 주문의 holding cost
- h : 완제품 재고 holding cost
- b : backlog cost
$$
Cost_t = c_1 q_t^N + c_2 q_t^E + h_1 N_1(t) + h_2 N_2(t) + h(IN(t))^+ + b(IN(t))^-
$$
1. 양의 재고
$$
(IN(t))^+ = \max(IN(t), 0)
$$
- IN(t)\>0 → 실제로 창고에 남아 있는 재고
- IN(t)\<0 → 재고는 없으므로 0
2. Backlog
$$
(IN(t))^- = \max(-IN(t), 0)
$$
- IN(t)\<0이면 아직 처리하지 못한 밀린 주문량
- IN(t)\>0이면 backlog는 0
- **Objective**
>
논문은 최적 주문 정책을 찾기 위해 두 가지 목적함수를 고려
- 두 가지 목적함수 :
1. Discounted cost
2. Long-run average cost
1. Discounted cost
- 미래 비용을 현재보다 조금 덜 중요하게 보는 방식
- 가까운 시점의 비용은 크게 반영
- 먼 미래의 비용은 γt만큼 할인해서 반영
- 예를 들어 γ=0.9라면,
- 지금 비용: Cost0
- 1번 뒤 비용: 0.9Cost1
- 2번 뒤 비용: (0.9)\^2Cost2 = 0.81Cost2
- 10번 뒤 비용: (0.9)\^10Cost10
- 0\<γ\<1
$$
\min_{\pi} E_{\pi} \left[\sum_{t=0}^{\infty} \gamma^t Cost_t \right]
$$
1. Long-run average cost
- 아주 긴 기간 동안 운영한다고 가정
- 전체 비용을 기간 수로 나누어 **기간당 평균 비용**을 최소화
- 예를 들어 100일 동안 총 비용이 1000이면,
- 1000/100 = 10
- 즉, 하루 평균 비용은 10
- 장기적으로 안정적인 운영 정책을 찾는 목적함수
$$
\min_{\pi} \limsup_{T \to \infty} \frac{1}{T} E_{\pi} \left[\sum_{t=0}^{T-1} Cost_t \right]
$$
---
## 논문 2. Dual Sourcing + Order Tracking + Uncertain Lead Times

---
### 1. 논문에서 다루는 재고관리 문제
- 단일 제품 재고관리 문제
- 두 개의 공급처를 사용함
<table header-row="true">

<tr>
<td>공급처</td>
<td>특징</td>
</tr>
<tr>
<td>Normal source</td>
<td>저렴하지만 느림</td>
</tr>
<tr>
<td>Emergency source</td>
<td>비싸지만 빠름</td>
</tr>
</table>
[image omitted: temporary Notion asset]
- Normal order는 두 단계의 공급 과정을 모두 거침
- Emergency order는 첫 번째 단계를 건너뛰고 두 번째 단계로 바로 들어감
- 각 단계의 처리시간은 Erlang distribution을 따름
- 주문이 현재 어느 단계까지 처리되었는지를 추적할 수 있음
- 수요는 Poisson process를 따름
- 미충족 수요는 두 방식으로 고려
- Backlogging
- Lost sales
### 2. 기존 toy model과의 차이
<table header-row="true">

<tr>
<td>구분</td>
<td>기존 toy model</td>
<td>이번 논문</td>
</tr>
<tr>
<td>공급처</td>
<td>1개</td>
<td>2개</td>
</tr>
<tr>
<td>주문 결정</td>
<td>주문량 (q_t) 하나</td>
<td>normal 주문 + emergency 주문</td>
</tr>
<tr>
<td>리드타임</td>
<td>없음 또는 단순 고정</td>
<td>queue 처리시간에 의해 확률적으로 결정</td>
</tr>
<tr>
<td>주문 추적</td>
<td>없음</td>
<td>주문의 처리 단계 추적</td>
</tr>
<tr>
<td>State</td>
<td>현재 재고 중심</td>
<td>재고 + server별 주문 진행 상태</td>
</tr>
<tr>
<td>수요</td>
<td>Uniform 또는 Poisson</td>
<td>Poisson process</td>
</tr>
<tr>
<td>재고 부족</td>
<td>주로 lost sales</td>
<td>backlogging과 lost sales 모두 고려</td>
</tr>
<tr>
<td>풀이 방법</td>
<td>RL 실험</td>
<td>DP로 최적정책 분석 + heuristic 제안</td>
</tr>
</table>
### 3. [논문 1]과 다른점 
#### Order Tracking
- 주문이 단순히 “아직 안 왔다”가 아니라, 어느 정도 처리되었는지를 추적함
- 예를 들어 normal order가 server 1에서 30% 처리되었는지, server 2에서 70% 처리되었는지를 state에 반영
- 이 정보가 있으면 남은 리드타임을 더 정확히 예측할 수 있음
- 중요한 이유 :
- 리드타임이 불확실할 때, 현재 재고만 보고 주문하면 부족함
- 이미 주문한 물량이 곧 도착할지, 한참 뒤에 도착할지에 따라 주문 결정이 달라져야 함
- 따라서 order tracking 정보는 더 정교한 주문 정책을 만드는 데 도움을 줌
### 4. MDP 환경 정의
- **State**
>
$$
s_t = (IN(t), W_2(t), N_2(t), W_1(t), N_1(t))
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>IN(t)</td>
<td>순재고 = 보유재고 - backlog</td>
</tr>
<tr>
<td>N_1(t)</td>
<td>server 1에 있는 outstanding order 수</td>
</tr>
<tr>
<td>N_2(t)</td>
<td>server 2에 있는 outstanding order 수</td>
</tr>
<tr>
<td>W_1(t)</td>
<td>server 1에서 현재 처리 중인 주문의 완료 비율</td>
</tr>
<tr>
<td>W_2(t)</td>
<td>server 2에서 현재 처리 중인 주문의 완료 비율</td>
</tr>
</table>
- W_1(t), W_2(t)가 바로 order tracking 정보
- 즉, 주문이 각 server에서 얼마나 진행되었는지를 나타냄
- **State 축소**
> 5차원 state를 그대로 쓰면 복잡하므로, 다음 3개 변수로 줄임
$$
X(t) = IN(t) + W_2(t)
$$
$$
Y(t) = N_2(t) - W_2(t)
$$
$$
Z(t) = N_1(t) - W_1(t)
$$
- 축소된 state :
$$
s_t = (X(t), Y(t), Z(t))
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>X(t)</td>
<td>현재 재고 가용성 + server 2에서 곧 도착할 정도</td>
</tr>
<tr>
<td>Y(t)</td>
<td>server 2의 workload</td>
</tr>
<tr>
<td>Z(t)</td>
<td>server 1의 workload</td>
</tr>
</table>
- X(t): 재고가 얼마나 여유 있는지
- Y(t): server 2가 얼마나 밀려 있는지
- Z(t): server 1이 얼마나 밀려 있는지
- **Action**
> 논문에서 action은 두 공급처에서 얼마나 주문할지를 결정하는 것
$$
a_t = (q_1, q_2)
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>(q_1)</td>
<td>normal source 주문량</td>
</tr>
<tr>
<td>(q_2)</td>
<td>emergency source 주문량</td>
</tr>
</table>
- q1: server 1로 들어가는 주문
- q2: server 1을 건너뛰고 server 2로 바로 들어가는 주문
- emergency source는 더 빠르지만 단위 주문 비용이 더 높음
- **Demand**
> 수요는 Poisson process를 따름
$$
D_t \sim Poisson(\lambda)
$$
- λ : 수요 도착률
- 수요가 발생하면 보유재고에서 차감
- 재고가 부족하면 backlogging 또는 lost sales로 처리
- **Lead Time**
>
🌟 이 논문에서 리드타임은 고정값이 아님
- Normal source:
- server 1 처리시간 + server 2 처리시간
[image omitted: temporary Notion asset]
- Emergency source:
- server 2 처리시간만 필요
각 server의 processing time은 Erlang distribution을 따름
- Erlang :  여러 개의 exp 처리 단계를 모두 통과해야 끝나는 시간
- 예를 들어 Server 1 작업이 한 번에 끝나는 게 아니라,
```plain text
1단계 → 2단계 → 3단계 → ... → r단계를 거쳐야 할때,
```
- 각 단계의 처리시간이 Exp 분포를 따른다면,
- 그 전체 처리시간은 Erlang 분포
<table header-row="true">

<tr>
<td>구분</td>
<td>Exp 분포</td>
<td>Erlang 분포</td>
</tr>
<tr>
<td>의미</td>
<td>한 단계 처리시간</td>
<td>여러 단계 전체 처리시간</td>
</tr>
<tr>
<td>단계 수</td>
<td>1개</td>
<td>r개</td>
</tr>
<tr>
<td>진행상태 추적</td>
<td>어려움</td>
<td>가능</td>
</tr>
<tr>
<td>예시</td>
<td>처리 하나가 랜덤하게 끝남</td>
<td>여러 공정 단계를 지나야 끝남</td>
</tr>
</table>
$$
T_1 \sim Erlang(r, \mu_1)
$$
$$
T_2 \sim Erlang(k, \mu_2)
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>r</td>
<td>server 1의 처리 단계 수</td>
</tr>
<tr>
<td>k</td>
<td>server 2의 처리 단계 수</td>
</tr>
<tr>
<td>mu_1</td>
<td>server 1의 처리율</td>
</tr>
<tr>
<td>mu_2</td>
<td>server 2의 처리율</td>
</tr>
</table>
- normal lead time = server 1에서 걸리는 시간 + server 2에서 걸리는 시간
- emergency lead time = server 2에서 걸리는 시간
- **Transition**
> 논문에서는 상태 전이가 사건 중심으로 일어남
- 주요 사건
1. 수요 발생
2. server 1 처리 진행
3. server 2 처리 진행
4. server 2 처리 완료 후 입고
1. **수요 발생**
수요가 발생하면 순재고가 감소
$$
IN(t+1) = IN(t) - 1
$$
- 재고가 있으면 재고 감소
- 재고가 없으면 backlog 증가 또는 lost sales 발생
2. **server 1 처리 진행**
server 1에서 processing phase가 하나 완료되면 W1(t)가 증가
$$
W_1(t) \rightarrow W_1(t) + \frac{1}{r}
$$
- r개의 phase 중 하나가 완료되었다는 의미
- 마지막 phase가 끝나면 해당 주문은 server 2로 이동
3. **server 2 처리 진행**
server 2에서 processing phase가 하나 완료되면 W2(t)가 증가
$$
W_2(t) \rightarrow W_2(t) + \frac{1}{k}
$$
- k개의 phase 중 하나가 완료되었다는 의미
- 마지막 phase가 끝나면 재고로 입고됨
4. **server 2 처리 완료 후 입고**
server 2 처리가 모두 끝나면 재고가 1개 증가
$$
IN(t+1) = IN(t) + 1
$$
- **Cost**
> 논문에서는 주문 비용, 재고 보유 비용, backlog 또는 shortage 비용을 고려
- **기본 비용식**
$$
c(X(t),Y(t),Z(t))
=
h\lfloor X(t)^+ \rfloor
+
b\lceil X(t)^- \rceil
+
h_2\lceil Y(t)\rceil
+
h_1\lceil Z(t)\rceil
$$
<table header-row="true">

<tr>
<td>항목</td>
<td>의미</td>
</tr>
<tr>
<td>h[X(t)\^+]</td>
<td>완제품 재고 보유 비용</td>
</tr>
<tr>
<td>b[X(t)\^-]</td>
<td>backlog 비용</td>
</tr>
<tr>
<td>h2[Y(t)]</td>
<td>server 2에 있는 주문의 holding cost</td>
</tr>
<tr>
<td>h2[Z(t)]</td>
<td>server 1에 있는 주문의 holding cost</td>
</tr>
</table>
- **Objective**
> 논문의 목표는 무한 기간에서 기대 할인 비용을 최소화하는 것
$$
\min_{\pi} E_{\pi}
\left[
\sum_{t=0}^{\infty}
\alpha^t Cost_t
\right]
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>pi</td>
<td>주문 정책</td>
</tr>
<tr>
<td>alpha</td>
<td>할인율</td>
</tr>
<tr>
<td>Cost_t</td>
<td>시점 (t)의 비용</td>
</tr>
</table>
- 현재부터 먼 미래까지 발생하는 비용을 할인해서 모두 더한 뒤,
- 그 기대값이 가장 작아지는 주문 정책을 찾는 것
---
## MDP 환경 정의
---
#### MDP 기본 가정
- 기간은 T일
- 매 기간 수요 dt가 확률적으로 발생
- 공급처는 2개
- 일반 공급처 R: 저렴하지만 리드타임이 김
- 긴급 공급처 E: 비싸지만 리드타임이 짧음
- 주문량은 바로 재고에 들어오지 않고, 리드타임 이후 입고
- 실제 리드타임은 고정값이 아니라 지연될 수 있음
- 기존 toy model과 맞추기 위해 backlog가 아니라 **결품/lost sales** 구조 사용
#### State
$$
s_t = (x_t, t, P_t^R, P_t^E)
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>xt</td>
<td>현재 재고량</td>
</tr>
<tr>
<td>t</td>
<td>현재 시점</td>
</tr>
<tr>
<td>Pt\^R</td>
<td>일반 공급처에서 주문했지만 아직 도착하지 않은 물량 정보</td>
</tr>
<tr>
<td>Pt\^E</td>
<td>긴급 공급처에서 주문했지만 아직 도착하지 않은 물량 정보</td>
</tr>
</table>
- Pt\^R, Pt\^E는 **pipeline inventory**
- pipeline inventory :  이미 주문은 했지만, 아직 창고 재고로 도착하지 않은 물량
- 현재 재고 + 아직 도착하지 않은 일반 주문 + 아직 도착하지 않은 긴급 주문
- 이 정보를 state에 넣는 이유 → 이미 주문한 물량이 미래에 도착해서 재고에 영향을 주기 때문
#### Pipeline 표현
조금 더 구체적으로 쓰면, 각 공급처의 pipeline을 벡터로 둘 수 있음
$$
P_t^R = (p_{t,1}^R, p_{t,2}^R, \dots, p_{t,L_{\max}^R}^R)
$$
$$
P_t^E = (p_{t,1}^E, p_{t,2}^E, \dots, p_{t,L_{\max}^E}^E)
$$
- 의미 :
- p_(t,1)\^R : 일반 공급처 주문 중 1기간 뒤 도착 예정 물량
- p_(t,2)\^R : 일반 공급처 주문 중 2기간 뒤 도착 예정 물량
- p_(t,1)\^E : 긴급 공급처 주문 중 1기간 뒤 도착 예정 물량
- 즉, pipeline은 “이미 주문했는데 아직 안 온 물량”을 도착 예정 시점별로 저장하는 구조
#### Action
$$
a_t = (q_t^R, q_t^E)
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>(q_t\^R)</td>
<td>일반 공급처 주문량</td>
</tr>
<tr>
<td>(q_t\^E)</td>
<td>긴급 공급처 주문량</td>
</tr>
</table>
- **주문량 제약**
- 기존 toy model의 주문 상한 M을 유지하면 다음과 같이 표현 가능
$$
q_t^R + q_t^E \le M
$$
$$
q_t^R, q_t^E \in \{0,1,\dots,M\}
$$
- 즉, 총 주문량은 최대 M개까지 가능하고, 그 안에서 일반 주문과 긴급 주문을 나누는 구조
#### Demand
$$
d_t \sim P_D
$$
1. Uniform
$$
d_t \sim Uniform(0,\lambda)
$$
2. Poisson
$$
d_t \sim Poisson(\lambda)
$$
#### Lead Time 불확실성
$$
L_t^R = \bar{L}^R + \Delta_t^R
$$
$$
L_t^E = \bar{L}^E + \Delta_t^E
$$
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>bar L\^R</td>
<td>일반 공급처의 기본 리드타임</td>
</tr>
<tr>
<td>bar L\^E</td>
<td>긴급 공급처의 기본 리드타임</td>
</tr>
<tr>
<td>Delta_t\^R</td>
<td>일반 공급처의 추가 지연</td>
</tr>
<tr>
<td>Delta_t\^E</td>
<td>긴급 공급처의 추가 지연</td>
</tr>
</table>
- 관계
$$
\bar{L}^R > \bar{L}^E
$$
- 일반 공급처는 싸지만 느림
- 긴급 공급처는 비싸지만 빠름
- 하지만 둘 다 지연될 수 있음
#### Arrival
시점 t에 실제로 도착하는 물량은 과거에 주문한 것 중 리드타임이 끝난 물량
- **일반 공급처 입고량**
$$
A_t^R = \sum_{\tau=0}^{t-1} q_{\tau}^R \cdot \mathbf{1}_{\{\tau + L_{\tau}^R = t\}}
$$
- **긴급 공급처 입고량**
$$
A_t^E = \sum_{\tau=0}^{t-1} q_{\tau}^E \cdot \mathbf{1}_{\{\tau + L_{\tau}^E = t\}}
$$
- **전체 입고량**
$$
A_t = A_t^R + A_t^E
$$
- 과거에 주문한 물량 중 오늘 도착할 차례가 된 것만 현재 재고에 더해짐
- τ는 **과거의 주문 시점**
- 즉 과거 시점 τ + Lead time = t (오늘) → 1 (오늘 입고됨)
#### Transition
- 입고 후 사용 가능한 재고
$$
I_t = x_t + A_t
$$
- 수요가 발생한 뒤 다음 재고
$$
x_{t+1} = \min(C, \max(I_t - d_t, 0))
$$
- 결품량
$$
z_t = \max(d_t - I_t, 0)
$$
- 여기서는 backlog가 아니라 lost sales 구조라서, 결품된 수요는 다음 기간으로 넘어가지 않고 비용으로만 반영 
<table header-row="true">

<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>It</td>
<td>수요 발생 전 사용 가능한 재고</td>
</tr>
<tr>
<td>x_\{t+1\}</td>
<td>다음 기간 재고</td>
</tr>
<tr>
<td>zt</td>
<td>수요를 만족하지 못한 결품량</td>
</tr>
<tr>
<td>C</td>
<td>창고 용량</td>
</tr>
</table>
#### Cost
$$
Cost_t =
K_R y_t^R + K_E y_t^E
+ c_R q_t^R + c_E q_t^E
+ h x_{t+1} + p z_t
$$
<table header-row="true">

<tr>
<td>항목</td>
<td>의미</td>
</tr>
<tr>
<td>K_R y_t\^R</td>
<td>일반 공급처 주문 고정비</td>
</tr>
<tr>
<td>K_E y_t\^E</td>
<td>긴급 공급처 주문 고정비</td>
</tr>
<tr>
<td>c_R q_t\^R</td>
<td>일반 공급처 단위 주문 비용</td>
</tr>
<tr>
<td>c_E q_t\^E</td>
<td>긴급 공급처 단위 주문 비용</td>
</tr>
<tr>
<td>h x_\{t+1\}</td>
<td>재고 유지 비용</td>
</tr>
<tr>
<td>p z_t</td>
<td>결품 비용</td>
</tr>
</table>
- 일반적으로 긴급 공급처가 더 비싸므로 
$$
c_E > c_R
$$
#### 주문 여부 변수
주문 고정비를 계산하려면 주문 여부 변수가 필요
→ 주문 고정비 : 주문을 한 번 넣을 때마다 무조건 발생하는 비용
- **일반 공급처 주문 여부**
$$
y_t^R =
\begin{cases}
1, & q_t^R > 0 \\
0, & q_t^R = 0
\end{cases}
$$
- **긴급 공급처 주문 여부**
$$
y_t^E =
\begin{cases}
1, & q_t^E > 0 \\
0, & q_t^E = 0
\end{cases}
$$
#### Reward
강화학습에서는 비용을 최소화하고 싶지만, 알고리즘은 reward를 최대화
$$
r_t = -Cost_t
$$
- 즉, 비용이 작을수록 reward가 커짐
#### Objective
- **비용 최소화 관점**
$$
\min_{\pi} E_{\pi}\left[\sum_{t=0}^{T-1} Cost_t\right]
$$
- **강화학습 reward 관점**
$$
\max_{\pi} E_{\pi}
\left[
\sum_{t=0}^{T-1} r_t
\right]
$$
- 총 비용 최소화 = 누적 reward 최대화
---

---

# 강화학습 결과 정리

---
## 목차
---
## 강화학습 모델 → **DQN 계열 모델**
---
- **DQN 계열 모델 선택 이유 :**
- 우리 문제의 행동이 연속값이 아니라, 
- 가능한 주문량 조합 중 하나를 선택하는 **이산 행동 문제**로 구성될 수 있기 때문
- 일반 공급처 주문량 qR과 긴급 공급처 주문량 qE를 동시에 결정해야 하므로, 두 주문량의 조합을 하나의 action으로 묶는 **Joint action 방식**을 사용
- 구현에서는 다음과 같이 action space를 구성 :
- q_R=0∼50
- q_E=0∼20
- 전체 action 수: 51×21=1071
- 즉, action index 하나가 특정 주문 조합 (qR,qE) 하나를 의미
---
## Double DQN
---
#### 모델 설명 
- Double DQN은 기본 DQN의 단점을 보완한 모델
- **기본 DQN **
- DQN은 현재 상태에서 각 action의 가치를 나타내는 Q(s,a)를 신경망으로 근사함
- 학습된 Q-value를 기준으로 가장 좋은 action을 선택함
- 하지만 특정 action의 가치를 실제보다 크게 추정하는 **overestimation bias** 문제가 발생할 수 있음
- **Double DQN**은 이를 줄이기 위해 action 선택과 action 평가를 분리
- Online network : 다음 상태에서 어떤 action이 좋은지 선택
- Target network : 선택된 action의 Q-value를 평가
- 이를 통해 Q-value가 과대평가되는 문제를 완화
#### 사용 이유
- 우리 문제는 수요와 리드타임이 확률적으로 변동하기 때문에, 같은 action을 선택해도 매 episode마다 비용이 달라질 수 있음
- 이런 stochastic 환경에서는 우연히 낮은 비용이 나온 action을 지나치게 좋게 평가할 위험 존재
- Double DQN은 Q-value 과대평가를 줄여주기 때문에, 기본 DQN보다 안정적인 학습을 기대할 수 있음
- 또한 action space가 1071개로 비교적 크기 때문에, Q-value 추정의 안정성이 중요
- 따라서 Double DQN을 첫 번째 강화학습 모델로 사용
#### 1. Learning Curve
[image omitted: temporary Notion asset]

- 이 그래프는 학습 episode가 증가함에 따라 Double DQN의 비용이 어떻게 변화하는지 보여주는 그래프
- Moving average가 전체적으로 감소하므로, Double DQN이 비용을 줄이는 방향으로 주문정책을 학습했음을 확인할 수 있음
- 학습 중간에 비용 spike가 발생하는 이유는 수요와 리드타임이 확률적으로 변동하기 때문
- 또한 학습 과정에서는 epsilon-greedy 탐색으로 랜덤 행동이 포함되기 때문에, 학습 비용은 평가 비용보다 크게 흔들릴 수 있음

#### 2. Trajectory
[image omitted: temporary Notion asset]

- 이 그래프는 특정 episode에서 Double DQN이 기간별로 어떤 주문을 하고, 그 결과 재고와 비용이 어떻게 변했는지 보여주는 그래프
- Double DQN은 일반 주문을 기본적으로 사용하고, 부족 위험이 있는 구간에서 긴급 주문을 함께 사용하는 패턴을 보임
- 초반에는 초기 재고와 pipeline이 비어 있어 shortage와 비용이 크게 발생함
- 이후에는 재고가 일정 수준으로 회복되면서 기간별 비용이 비교적 안정되는 흐름을 보임

#### 3. Average Cost
[image omitted: temporary Notion asset]

- 이 그래프는 각 정책의 총비용이 어떤 비용 항목에서 발생했는지 보여주는 그래프
- Double DQN은 TBS보다 낮은 총비용을 보였지만, Single-Index와 Dual-Index보다는 높은 비용을 기록
- Double DQN은 일반 주문과 긴급 주문을 함께 사용하는 전략을 학습했지만, 보유비용과 백오더 비용을 완전히 균형 있게 줄이지는 못함
- 기존 휴리스틱은 재고관리 구조를 반영한 정책이기 때문에 비용 항목 간 균형이 더 안정적으로 나타난 것으로 해석할 수 있음

#### 4. Mean Cost
[image omitted: temporary Notion asset]

- 이 그래프는 정책별 평균 총비용을 직접 비교하여 어떤 정책이 더 효율적인지 보여주는 그래프
- Double DQN은 TBS보다 좋은 성능을 보였지만, Single-Index와 Dual-Index보다는 약간 높은 비용을 기록
- Double DQN은 사전에 정해진 주문 규칙 없이 경험을 통해 정책을 학습했기 때문에 기존 휴리스틱에 근접한 결과를 얻은 것으로 볼 수 있음
- 그러나 Single-Index와 Dual-Index는 grid search로 최적 파라미터를 찾은 구조적 정책이므로, 현재 Double DQN보다 더 안정적인 성능을 보인 것으로 해석됨

---
## Dueling Double DQN
---
#### 모델 개요
- **Dueling Double DQN** :  Double DQN에 **Dueling Network 구조**를 추가한 모델
- 일반 DQN은 각 action의 Q-value를 바로 예측
- 반면 Dueling 구조는 Q-value를 두 부분으로 나누어 학습
- V(s) : 현재 상태 자체의 가치
- A(s,a) : 해당 상태에서 특정 action이 평균보다 얼마나 좋은지
- 즉, 상태의 좋고 나쁨과 action의 상대적 효과를 분리해서 학습
#### 사용 이유
- 우리 문제에서는 어떤 상태 자체가 이미 좋은 상태이거나 나쁜 상태일 수 있음
- 예 :  재고가 충분하고 도착 예정 물량이 많으면 비교적 좋은 상태
- 예 :  재고가 부족하고 pipeline도 부족하면 위험한 상태
- 이런 경우 모든 action의 가치를 처음부터 따로 학습하는 것보다, 상태 자체의 가치를 먼저 구분하는 것이 효율적
- 특히 우리 문제는 action 수가 1071개로 많기 때문에, 모든 action 조합의 Q-value를 안정적으로 학습하기 어려움
- Dueling 구조는 상태 가치와 action 효과를 분리하므로, action space가 큰 문제에서 더 효율적인 학습을 기대할 수 있음
- 따라서 Double DQN보다 더 안정적이고 세밀한 주문정책을 학습할 가능성이 있어 사용
#### 1. Learning Curve
[image omitted: temporary Notion asset]

- 이 그래프는 학습이 진행되면서 Dueling Double DQN의 비용이 어떻게 변하는지 보여주는 그래프
- Moving average가 빠르게 감소한 뒤 후반부에는 안정적으로 유지되는 흐름을 보임
- 이는 Dueling Double DQN이 학습 초반에는 탐색을 많이 하다가, 점차 비용이 낮은 주문정책을 학습했음을 의미
- 후반부에도 일부 spike가 존재하지만, 전체적으로는 Double DQN보다 더 안정적인 학습 흐름을 보임

#### 2. Trajectory
[image omitted: temporary Notion asset]

- 이 그래프는 특정 episode에서 Dueling Double DQN이 기간별로 어떤 주문을 하고, 재고와 비용이 어떻게 변했는지 보여주는 그래프
- Dueling Double DQN은 일반 주문을 기본적으로 사용하고, 필요한 시점에 긴급 주문을 보조적으로 사용하는 패턴을 보임
- 초반에는 초기 재고와 pipeline이 비어 있어 shortage가 크게 발생하지만, 이후에는 재고가 일정 범위에서 회복
- 전체 episode 비용이 약 25,159로 나타나며, 학습된 정책이 수요 변동에 대응하는 주문 패턴을 형성했음을 보여줌

#### 3. Average Cost
[image omitted: temporary Notion asset]

- 이 그래프는 각 정책의 평균 비용이 주문비용, 보유비용, 백오더 비용 중 어디에서 발생했는지 보여주는 그래프
- Dueling Double DQN은 MIP Lower Bound를 제외하면 가장 낮은 총비용을 기록
- Dueling Double DQN은 일반 주문과 긴급 주문을 함께 사용하면서도, 전체 비용을 기존 휴리스틱보다 낮게 유지
- 이는 상태 가치와 행동 이점을 분리해 학습한 구조가 주문 조합 선택에 효과적으로 작동한 결과로 볼 수 있음

#### 4. Mean Cost
[image omitted: temporary Notion asset]

- 이 그래프는 정책별 평균 총비용을 비교하여 어떤 정책이 가장 효율적인지 보여주는 그래프
- Dueling Double DQN은 평균 비용 약 23,521로, Single-Index, Dual-Index, TBS보다 낮은 비용을 기록
- 기존 휴리스틱은 고정된 정책 구조를 사용하지만, Dueling Double DQN은 상태에 따라 주문 조합을 유연하게 선택
- MIP Lower Bound는 미래 정보를 알고 계산한 이상적 기준이므로, 실제 정책인 Dueling Double DQN보다 낮게 나오는 것이 정상

---

---

# 모델 정리

---
## 목차
---
## Double DQN 구조
---
#### **네트워크 구조 (Q-Network)**
- **Input → Linear(13 → 256) → ReLU**
- **Linear(256 → 256) → ReLU**
- **Linear(256 → 256) → ReLU**
- **Output → Linear(256 → 1,071) [Q-value]**
- Input 차원: 13
- 현재 재고 + 일반 공급처 pipeline + 긴급 공급처 pipeline
- Output 차원: 1,071
- q_R = 0\~50,  q_E =0\~2
- 총 51 x 21 = 1,071개의 joint action
#### **하이퍼파라미터**
- Learning rate: **1e-3 (Adam)**
- 할인율: **0.99**
- Replay buffer 크기: **100,000**
- Batch size: **256**
- Target update: **1,000 step마다**
- epsilon 초기값: **1.0**
- epsilon 최솟값: **0.05**
- epsilon 감소 step: **600,000**
- Gradient clip: **10.0**
- Reward scale: **÷ 1,000**
- 총 학습 에피소드: **10,000**
#### **학습 방식**
- 1개의 환경에서 episode 단위로 경험을 수집함
- epsilon-greedy 방식으로 action을 선택함
- 초반에는 랜덤 action을 많이 선택하여 탐색함
- 학습이 진행될수록 Q-value가 높은 action을 선택함
- Replay buffer에 transition을 저장함
- (state, action, reward, next\\ state, done)
- Buffer에서 mini-batch를 샘플링하여 Q-network를 업데이트함
- Double DQN 방식으로 action 선택과 평가를 분리함
- Online network: 다음 상태에서 action 선택
- Target network: 선택된 action의 Q-value 평가
- Loss는 Smooth L1 Loss(Huber loss)를 사용함
- Reward는 비용 최소화를 위해 -cost / 1000으로 사용함
---
## **Dueling Double DQN 구조**
---
#### **네트워크 구조 (Dueling Q-Network)**
- **Shared Feature Network**
- **Input → Linear(13 → 256) → ReLU**
- **Linear(256 → 256) → ReLU**
- **Value Stream**
- **Linear(256 → 256) → ReLU**
- **Linear(256 → 1) [V(s)]**
- **Advantage Stream**
- **Linear(256 → 256) → ReLU**
- **Linear(256 → 1,071) [A(s,a)]**
- **Q-value 계산**
- Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
- Input 차원: 13
- Output 차원: 1,071
- 상태 자체의 가치와 action별 이점을 나누어 학습함
#### **하이퍼파라미터**
- Learning rate: **1e-3 (Adam)**
- 할인율: **0.99**
- Replay buffer 크기: **100,000**
- Batch size: **256**
- Target update: **1,000 step마다**
- epsilon 초기값: **1.0**
- epsilon 최솟값: **0.05**
- epsilon 감소 step: **600,000**
- Gradient clip: **10.0**
- Reward scale: **÷ 1,000**
- 총 학습 에피소드: **10,000**
#### **학습 방식**
- Double DQN과 동일하게 replay buffer 기반으로 학습함
- epsilon-greedy 방식으로 action을 선택함
- Online network와 Target network를 사용하여 Double DQN 업데이트를 수행함
- Dueling 구조를 통해 상태 가치와 action 이점을 분리해서 학습함
- 상태 자체가 좋은지 나쁜지를 먼저 판단하고, 그 상태에서 어떤 주문 조합이 더 좋은지 학습함
- Action space가 1,071개로 크기 때문에, Dueling 구조가 많은 action 중 적절한 주문 조합을 선택하는 데 유리함
- Reward는 비용 최소화를 위해 -cost / 1000으로 사용함
---

---

# 01] Course introduction_organized

---
## 목차
---
<pdf src="file://%7B%22source%22%3A%22attachment%3A6550db99-aa07-452b-84e4-3a717f5c442b%3A01Course_introduction_organized.pdf%22%2C%22permissionRecord%22%3A%7B%22table%22%3A%22block%22%2C%22id%22%3A%2237e20172-a832-806d-a4a4-dc2c134096eb%22%2C%22spaceId%22%3A%229fcf7087-7108-4498-a15f-963affc7fe7d%22%7D%7D"></pdf>
## 1. 꼭 알아야 할 개념 구분
### Supervised / Unsupervised / Reinforcement Learning
<table header-row="true">
<tr>
<td>구분</td>
<td>핵심 의미</td>
<td>데이터 형태</td>
</tr>
<tr>
<td>**Supervised learning**</td>
<td>정답이 있는 데이터를 보고 학습</td>
<td>x,yx, yx,y</td>
</tr>
<tr>
<td>**Unsupervised learning**</td>
<td>정답 없이 데이터 속 패턴을 찾음</td>
<td>xxx</td>
</tr>
<tr>
<td>**Reinforcement learning**</td>
<td>환경과 상호작용하면서 좋은 행동을 학습</td>
<td>s,a,r,s′s, a, r, s's,a,r,s′</td>
</tr>
</table>
시험 포인트:
> 강화학습은 labeled data를 이용해 정답을 맞히는 방식이다.
→ **False**
> 강화학습은 state, action, reward, next state 경험을 이용한다.
→ **True**
---
## 2. 강화학습의 핵심 아이디어
강화학습은 **불확실한 환경에서 경험을 통해 좋은 의사결정을 배우는 방법**이야.
핵심 문장:
> Reinforcement learning은 agent가 environment와 상호작용하면서 cumulative reward를 최대화하는 policy를 학습하는 방법이다.
중요 키워드:
<table header-row="true">
<tr>
<td>개념</td>
<td>뜻</td>
</tr>
<tr>
<td>**Agent**</td>
<td>행동을 선택하는 주체</td>
</tr>
<tr>
<td>**Environment**</td>
<td>agent가 상호작용하는 대상</td>
</tr>
<tr>
<td>**State**</td>
<td>현재 상황</td>
</tr>
<tr>
<td>**Action**</td>
<td>agent가 선택하는 행동</td>
</tr>
<tr>
<td>**Reward**</td>
<td>행동 결과로 받는 보상</td>
</tr>
<tr>
<td>**Trial-and-error**</td>
<td>시행착오</td>
</tr>
<tr>
<td>**Delayed reward**</td>
<td>행동의 결과가 나중에 나타남</td>
</tr>
</table>
---
## 3. RL 구성요소
### Policy
**Policy는 상태를 보고 어떤 행동을 할지 정하는 규칙**이야.
예시:
> 재고가 적으면 주문을 많이 한다.
재고가 많으면 주문을 적게 한다.
정책은 두 종류가 가능해.
<table header-row="true">

<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>**Deterministic policy**</td>
<td>같은 상태에서 항상 같은 행동</td>
</tr>
<tr>
<td>**Stochastic policy**</td>
<td>같은 상태에서도 확률적으로 행동 선택</td>
</tr>
</table>
---
### Reward
**Reward는 행동이 좋았는지 나빴는지를 알려주는 신호**야.
강화학습의 목표는 단순히 지금 reward만 크게 만드는 게 아니라,
**장기적인 누적 reward를 최대화하는 것**이야.
---
### Value function
**Value function은 어떤 상태가 장기적으로 얼마나 좋은지를 나타내는 함수**야.
중요 구분:
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Reward**</td>
<td>지금 당장의 좋고 나쁨</td>
</tr>
<tr>
<td>**Value function**</td>
<td>미래까지 고려한 좋고 나쁨</td>
</tr>
</table>
---
### Model of environment
**Model은 환경이 어떻게 움직이는지에 대한 정보**야.
예를 들어 어떤 action을 했을 때 다음 state와 reward를 예측할 수 있으면 model이 있는 거야.
하지만 대부분의 RL은 **model-free** 방식이야.
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>**Model-based**</td>
<td>환경 모델을 알고 예측하면서 학습</td>
</tr>
<tr>
<td>**Model-free**</td>
<td>직접 경험하면서 시행착오로 학습</td>
</tr>
</table>
---
## 4. RL 방법의 종류
### Tabular methods
상태와 행동의 수가 작으면 표로 value function을 저장할 수 있어.
대표 예시:
- SARSA
- Q-learning
핵심:
> 작은 문제에서는 Q-table 같은 표를 사용할 수 있다.
---
### Approximation methods
상태와 행동의 수가 너무 크면 표로 저장하기 어렵기 때문에,
value function이나 policy를 함수로 근사해.
특히 Deep RL은 neural network를 사용해 근사해.
대표 예시:
- DQN
- REINFORCE
- Actor-Critic
핵심:
> 큰 문제에서는 value function이나 policy를 neural network로 근사한다.
---
## 5. Optimization과 RL 차이
<table header-row="true">
<tr>
<td>구분</td>
<td>Optimization</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>방식</td>
<td>수학 모델을 세우고 최적해를 구함</td>
<td>환경과 상호작용하면서 정책을 학습</td>
</tr>
<tr>
<td>필요한 것</td>
<td>목적함수, 제약식, 시스템 모델</td>
<td>state, action, reward 경험</td>
</tr>
<tr>
<td>가정</td>
<td>시스템 모델이 알려져 있는 경우가 많음</td>
<td>모델을 몰라도 학습 가능</td>
</tr>
<tr>
<td>예시</td>
<td>LP, IP, stochastic programming, dynamic programming</td>
<td>Q-learning, DQN, REINFORCE, Actor-Critic</td>
</tr>
</table>
핵심 문장:
> Optimization은 보통 시스템 모델이 알려져 있다고 가정하고 최적해를 계산한다.
RL은 환경과 상호작용하면서 decision policy를 학습한다.
---
## 6. True/False 대비 문장
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>강화학습은 환경과 상호작용하면서 학습한다.</td>
<td>True</td>
</tr>
<tr>
<td>강화학습의 목표는 장기적인 누적 보상을 최대화하는 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>Policy는 state를 action으로 연결하는 규칙이다.</td>
<td>True</td>
</tr>
<tr>
<td>Reward는 항상 미래까지 고려한 가치이다.</td>
<td>False</td>
</tr>
<tr>
<td>Value function은 장기적인 관점에서 상태가 얼마나 좋은지 나타낸다.</td>
<td>True</td>
</tr>
<tr>
<td>Model-free RL은 환경 모델을 정확히 알고 있어야만 가능하다.</td>
<td>False</td>
</tr>
<tr>
<td>Tabular method는 상태와 행동 공간이 작을 때 적합하다.</td>
<td>True</td>
</tr>
<tr>
<td>DQN, REINFORCE, Actor-Critic은 approximation method에 해당한다.</td>
<td>True</td>
</tr>
<tr>
<td>Optimization은 보통 시스템 모델이 알려져 있다고 가정한다.</td>
<td>True</td>
</tr>
<tr>
<td>RL은 복잡하거나 모델링하기 어려운 문제에 실용적인 해를 제공할 수 있다.</td>
<td>True</td>
</tr>
</table>

---

# 02] Linear programming

---
## 목차
---
<pdf src="file://%7B%22source%22%3A%22attachment%3Ac1fd6223-5722-4b95-9a58-847293841bce%3A02Linear_programming.pdf%22%2C%22permissionRecord%22%3A%7B%22table%22%3A%22block%22%2C%22id%22%3A%2237e20172-a832-8081-81da-dfdd2008eedc%22%2C%22spaceId%22%3A%229fcf7087-7108-4498-a15f-963affc7fe7d%22%7D%7D"></pdf>
## 1. 개념 구분
### Decision variable, 의사결정변수
**내가 직접 결정해야 하는 값**이야.
예시:
- 제품을 몇 개 생산할지
- 공장에서 창고로 얼마나 보낼지
- 각 원료를 몇 % 섞을지
- 각 기간에 얼마나 생산할지
---
### Objective function, 목적함수
**최대화하거나 최소화하고 싶은 목표**야.
예시:
- 이익 최대화
- 비용 최소화
- 운송비 최소화
- 생산비 + 재고비 최소화
---
### Constraint, 제약식
**반드시 지켜야 하는 조건**이야.
예시:
- 원자재 사용량은 보유량을 넘으면 안 됨
- 생산량은 0 이상이어야 함
- 공장 생산량은 capacity를 넘으면 안 됨
- 창고 수요는 반드시 만족해야 함
- 재고는 0 이상이어야 함
---
## 2. LP / IP / MIP 구분
<table header-row="true">

<tr>
<td>개념</td>
<td>뜻</td>
<td>예시</td>
</tr>
<tr>
<td>**LP**</td>
<td>변수들이 연속형인 선형 최적화 문제</td>
<td>생산량이 3.5처럼 가능</td>
</tr>
<tr>
<td>**IP**</td>
<td>변수들이 정수형인 최적화 문제</td>
<td>제품 개수, 사람 수처럼 정수</td>
</tr>
<tr>
<td>**MIP**</td>
<td>연속형 변수와 정수형 변수가 섞인 문제</td>
<td>공장 열지 말지 + 물량 얼마나 보낼지</td>
</tr>
</table>
중요한 문장:
> **LP는 목적함수와 제약식이 선형이고, 의사결정변수가 연속형인 문제**이다.
---
## 3. Deterministic / Stochastic 구분
<table header-row="true">

<tr>
<td>개념</td>
<td>뜻</td>
<td>예시</td>
</tr>
<tr>
<td>**Deterministic model**</td>
<td>모든 값이 이미 알려져 있고 고정됨</td>
<td>수요가 500개로 주어짐</td>
</tr>
<tr>
<td>**Stochastic model**</td>
<td>일부 값이 확률변수임</td>
<td>수요가 확률분포를 따름</td>
</tr>
</table>
중요한 문장:
> 수요처럼 일부 파라미터가 랜덤이면 stochastic model이다.
---
## 4. 예제 구조 이해
### 생산 계획 문제
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>제품별 생산량</td>
</tr>
<tr>
<td>목적</td>
<td>이익 최대화</td>
</tr>
<tr>
<td>제약</td>
<td>원자재 사용량 제한, 생산량 0 이상</td>
</tr>
</table>
---
### 수송 문제
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>공장에서 창고로 보내는 양</td>
</tr>
<tr>
<td>목적</td>
<td>총 운송비 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>공장 공급량 제한, 창고 수요 만족, 운송량 0 이상</td>
</tr>
</table>
---
### 배합 문제
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>각 광석을 섞는 비율</td>
</tr>
<tr>
<td>목적</td>
<td>생산 비용 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>금속 성분 비율 조건, 전체 비율 합은 1</td>
</tr>
</table>
---
### 확률적 수요 문제
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>제품별 생산량</td>
</tr>
<tr>
<td>목적</td>
<td>기대 이익 최대화</td>
</tr>
<tr>
<td>특징</td>
<td>수요가 확률변수</td>
</tr>
<tr>
<td>핵심</td>
<td>너무 많이 만들면 재고/잔존가치 문제, 너무 적게 만들면 판매 기회 손실</td>
</tr>
</table>
---
### 다기간 생산 계획
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>기간별·공장별 생산량</td>
</tr>
<tr>
<td>목적</td>
<td>생산비 + 재고비 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>공장 capacity 제한, 재고 균형, 품절 불가</td>
</tr>
</table>
## 5. True/False 대비 문장
이 정도 문장만 판단할 수 있으면 충분해.
<table header-row="true">

<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>최적화 문제는 의사결정변수, 목적함수, 제약식으로 구성된다.</td>
<td>True</td>
</tr>
<tr>
<td>목적함수는 반드시 최대화 문제만 가능하다.</td>
<td>False</td>
</tr>
<tr>
<td>LP에서 모든 의사결정변수는 정수형이어야 한다.</td>
<td>False</td>
</tr>
<tr>
<td>MIP는 연속형 변수와 정수형 변수가 섞인 문제이다.</td>
<td>True</td>
</tr>
<tr>
<td>deterministic model은 **모든 파라미터가 알려져 있고 고정된 모델**이다.</td>
<td>True</td>
</tr>
<tr>
<td>stochastic model은 **일부 파라미터**가 확률변수인 모델이다.</td>
<td>True</td>
</tr>
<tr>
<td>생산 계획 문제에서 제품별 생산량은 의사결정변수이다.</td>
<td>True</td>
</tr>
<tr>
<td>수송 문제의 목적은 총 운송 비용을 최소화하는 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>배합 문제에서 각 원료의 비율 합은 1이 되어야 한다.</td>
<td>True</td>
</tr>
<tr>
<td>다기간 생산계획에서는 재고 균형 제약이 중요하다.</td>
<td>True</td>
</tr>
</table>

---

# 03] Mixed Integer Programming

---
## 목차
---
## 1. 이 PDF의 큰 주제
이 자료는 한마디로:
> **정수변수 또는 이진변수가 들어가는 최적화 문제를 어떻게 이해하고 모델링하는가**
를 다루는 자료야.
LP에서는 변수가 연속형이었다면,
MIP/IP에서는 **정수형 변수** 또는 **0/1 변수**가 등장해.
---
# 2. 기본 개념 구분
## 2-1. IP와 MIP
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
<td>핵심</td>
</tr>
<tr>
<td>**IP**</td>
<td>Integer Programming</td>
<td>모든 변수가 정수형</td>
</tr>
<tr>
<td>**MIP**</td>
<td>Mixed-Integer Programming</td>
<td>정수형 변수와 연속형 변수가 섞임</td>
</tr>
<tr>
<td>**LP**</td>
<td>Linear Programming</td>
<td>모든 변수가 연속형</td>
</tr>
</table>
시험용 문장:
> **MIP는 목적함수와 제약식은 선형이지만, 일부 의사결정변수가 정수형 또는 이진형인 문제이다.**
---
## 2-2. MIP/IP가 어려운 이유
자료에서 중요한 표현:
> **No known polynomial-time algorithm for solving general MIPs/IPs**
쉽게 말하면:
> 일반적인 MIP/IP 문제는 빠르게 풀 수 있는 보편적인 알고리즘이 알려져 있지 않다.
그래서 **LP relaxation**, **Branch-and-Bound**, **solver** 같은 개념이 필요해.
---
# 3. LP Relaxation
## 3-1. 의미
**LP relaxation**은 정수 조건을 풀어버리는 거야.
예를 들어 원래는:
> 변수는 정수여야 한다.
였는데, relaxation에서는:
> 변수는 실수여도 된다.
로 바꾸는 것.
즉,
> **MIP/IP를 더 쉬운 LP 문제로 완화해서 푸는 방법**
---
## 3-2. 왜 쓰는가?
LP relaxation은 정답을 바로 주기보다는 **기준값**, 즉 bound를 줘.
<table header-row="true">
<tr>
<td>문제 유형</td>
<td>LP relaxation이 주는 값</td>
</tr>
<tr>
<td>최대화 문제</td>
<td>Upper bound</td>
</tr>
<tr>
<td>최소화 문제</td>
<td>Lower bound</td>
</tr>
</table>
시험용 문장:
> **LP relaxation은 MIP보다 풀기 쉽고, MIP 해의 기준이 되는 bound를 제공한다.**
---
## 3-3. 주의점
자료에서 중요한 표현:
> **Rounding LP solution is not a good strategy**
즉,
> LP relaxation 해를 반올림한다고 해서 항상 정수계획 문제의 최적해가 되는 것은 아니다.
시험 포인트:
> “LP relaxation 해를 반올림하면 항상 최적해이다.”
→ **False**
---
# 4. Formulation이 중요한 이유
같은 MIP 문제라도 여러 방식으로 모델링할 수 있어.
중요한 건:
> **제약식 개수가 적다고 항상 좋은 모델은 아니다.**
자료의 핵심 표현:
> **What matters is how tight the LP relaxation is.**
즉,
> 좋은 formulation은 LP relaxation이 더 tight해서 더 좋은 bound를 제공하는 모델이다.
시험 포인트:
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>제약식 수가 적은 formulation이 항상 좋다.</td>
<td>False</td>
</tr>
<tr>
<td>Tight한 LP relaxation은 좋은 bound를 줄 수 있다.</td>
<td>True</td>
</tr>
</table>
---
# 5. Branch-and-Bound, B&B
자료에서 반드시 기억할 표현:
> **B&B ⇒ LP relaxation + Tree search**
쉽게 말하면:
> Branch-and-Bound는 LP relaxation을 풀면서, 정수 조건을 만족하도록 경우를 나누어 트리 형태로 탐색하는 방법이다.
너는 세부 계산을 외울 필요는 없어.
시험용으로는 이 정도만 알면 돼.
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Branch**</td>
<td>경우를 나누는 것</td>
</tr>
<tr>
<td>**Bound**</td>
<td>LP relaxation으로 기준값을 계산하는 것</td>
</tr>
<tr>
<td>**Tree search**</td>
<td>나눈 경우들을 트리처럼 탐색하는 것</td>
</tr>
</table>
핵심 문장:
> **B&B는 LP relaxation과 tree search를 결합한 MIP solver의 대표적 방법이다.**
---
# 6. 예제별 구조 이해
## 6-1. Facility Location Problem
시설 위치 결정 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>시설을 열지 말지, 고객을 어느 시설에 배정할지</td>
</tr>
<tr>
<td>목적</td>
<td>시설 개설 비용 + 고객 서비스 비용 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>모든 고객은 서비스받아야 함, 열린 시설만 고객을 서비스할 수 있음</td>
</tr>
</table>
핵심:
> 시설을 열지 말지는 0/1 변수이므로 MIP가 된다.
---
## 6-2. Capacitated Facility Location Problem
Facility Location에 **capacity 제한**이 추가된 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>시설 개설 여부, 고객 배정량</td>
</tr>
<tr>
<td>목적</td>
<td>총비용 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>고객 수요 만족, 시설 capacity 초과 불가</td>
</tr>
</table>
핵심:
> 각 시설이 처리할 수 있는 용량이 제한되어 있다.
---
## 6-3. Knapsack Problem
배낭 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>각 물건을 선택할지 말지</td>
</tr>
<tr>
<td>목적</td>
<td>선택한 물건의 총가치 최대화</td>
</tr>
<tr>
<td>제약</td>
<td>총무게가 배낭의 무게 제한을 넘으면 안 됨</td>
</tr>
</table>
핵심:
> 물건 선택 여부가 0/1 변수이므로 0-1 정수계획 문제이다.
확장:
<table header-row="true">
<tr>
<td>확장</td>
<td>의미</td>
</tr>
<tr>
<td>Multiple knapsack</td>
<td>배낭이 여러 개</td>
</tr>
<tr>
<td>Bounded knapsack</td>
<td>물건을 여러 개 선택 가능하지만 개수 제한 있음</td>
</tr>
<tr>
<td>Fractional knapsack</td>
<td>물건을 쪼개서 일부만 선택 가능</td>
</tr>
</table>
---
## 6-4. Assignment Problem
배정 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>사람과 작업을 연결할지 말지</td>
</tr>
<tr>
<td>목적</td>
<td>총 배정 비용 최소화 또는 가치 최대화</td>
</tr>
<tr>
<td>제약</td>
<td>각 사람은 하나의 작업, 각 작업도 하나의 사람에게 배정</td>
</tr>
</table>
핵심:
> Assignment problem은 **일대일 매칭 문제**이다.
---
## 6-5. Generalized Assignment Problem, GAP
Assignment problem의 확장 버전이야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>작업을 어떤 사람에게 배정할지</td>
</tr>
<tr>
<td>목적</td>
<td>총가치 최대화</td>
</tr>
<tr>
<td>제약</td>
<td>각 사람의 capacity 제한, 각 작업은 한 사람에게 배정</td>
</tr>
</table>
핵심:
> 한 사람이 여러 작업을 맡을 수 있지만, capacity를 넘으면 안 된다.
자료에서 중요한 관계:
<table header-row="true">
<tr>
<td>조건</td>
<td>연결되는 문제</td>
</tr>
<tr>
<td>직원이 1명뿐이면</td>
<td>Knapsack problem</td>
</tr>
<tr>
<td>직원 수와 작업 수가 같고, 한 직원이 하나의 작업만 맡으면</td>
<td>Assignment problem</td>
</tr>
</table>
---
## 6-6. Traveling Salesman Problem, TSP
외판원 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>어떤 도시에서 어떤 도시로 이동할지</td>
</tr>
<tr>
<td>목적</td>
<td>전체 이동거리 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>모든 도시를 한 번씩 방문하고 출발점으로 돌아옴</td>
</tr>
</table>
핵심:
> TSP는 모든 도시를 정확히 한 번씩 방문하고 다시 출발점으로 돌아오는 최단 경로 문제이다.
자료에서 중요한 표현:
> 가능한 tour 수는 (n−1)!(n-1)!(n−1)!개라서 brute-force가 어렵다.
그래서 실제로는 heuristic이나 meta-heuristic도 사용한다.
예시:
- Nearest neighbor search
- k-opt method
- Genetic algorithm
- RL
---
## 6-7. Subtour Elimination
TSP에서 매우 중요한 개념이야.
### Subtour란?
전체 도시를 하나로 연결해서 도는 게 아니라,
일부 도시끼리만 따로 순환하는 잘못된 경로야.
### Subtour elimination이란?
> 부분 순환이 생기지 않게 막는 제약
시험용 문장:
> **Subtour elimination은 TSP에서 전체 tour가 하나로 연결되도록 하고, 부분 tour를 제거하기 위한 제약이다.**
---
## 6-8. DFJ vs MTZ
TSP formulation 두 가지야.
<table header-row="true">
<tr>
<td>구분</td>
<td>특징</td>
</tr>
<tr>
<td>**DFJ**</td>
<td>더 강한 LP relaxation 제공</td>
</tr>
<tr>
<td>**MTZ**</td>
<td>ordering variable을 사용, 더 scalable</td>
</tr>
</table>
자료의 핵심 문장:
> **DFJ is stronger, while MTZ is more scalable.**
뜻:
> DFJ는 더 강한 formulation이지만 제약식이 많고, MTZ는 상대적으로 풀기 쉽고 확장성이 좋다.
---
## 6-9. Symmetric Solution Issue
TSP에서는 같은 경로라도 시작 도시를 다르게 쓰면 다른 해처럼 보일 수 있어.
예를 들어:
> 1 → 2 → 4 → 6 → 7 → 5 → 3 → 1
2 → 4 → 6 → 7 → 5 → 3 → 1 → 2
이 둘은 사실 같은 tour야.
핵심:
> TSP에는 symmetric solution issue가 있다.
---
## 6-10. Multi-period Production Planning with Setup Cost
여러 기간 동안 생산량을 결정하는 문제야.
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>기간별 생산량, 생산 여부, 재고 수준</td>
</tr>
<tr>
<td>목적</td>
<td>재고비 + setup cost 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>생산하지 않으면 생산량 0, 재고 균형 만족</td>
</tr>
</table>
핵심:
> 생산 여부는 0/1 변수이고 생산량은 연속변수이므로 MIP가 된다.
---
## 6-11. Big-M
자료 마지막의 중요한 개념이야.
### Big-M이란?
> 충분히 큰 수
### 역할
이진변수와 연속변수를 연결하는 데 사용돼.
예를 들어:
<table header-row="true">
<tr>
<td>생산 여부</td>
<td>의미</td>
</tr>
<tr>
<td>zt=0z_t = 0zt=0</td>
<td>생산하지 않음 → 생산량 xt=0x_t = 0xt=0</td>
</tr>
<tr>
<td>zt=1z_t = 1zt=1</td>
<td>생산함 → 생산량 허용</td>
</tr>
</table>
시험용 문장:
> **Big-M은 이진변수와 연속변수를 연결할 때 사용된다.**
---
# 7. True/False 대비
<table header-row="true">

<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>MIP는 일부 변수가 정수형이고 일부 변수가 연속형일 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>IP는 모든 변수가 연속형인 문제이다.</td>
<td>False</td>
</tr>
<tr>
<td>LP relaxation은 정수 조건을 완화하여 LP 문제로 바꾸는 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>LP relaxation 해를 반올림하면 항상 최적해가 된다.</td>
<td>False</td>
</tr>
<tr>
<td>B&B는 LP relaxation과 tree search를 결합한 방법이다.</td>
<td>True</td>
</tr>
<tr>
<td>좋은 formulation은 tight한 LP relaxation을 제공할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>제약식 수가 적은 formulation이 항상 좋은 formulation이다.</td>
<td>False</td>
</tr>
<tr>
<td>Knapsack problem은 물건을 선택할지 말지를 결정한다.</td>
<td>True</td>
</tr>
<tr>
<td>Assignment problem은 일대일 배정 문제이다.</td>
<td>True</td>
</tr>
<tr>
<td>GAP에서는 한 사람이 여러 작업을 맡을 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>TSP는 모든 도시를 한 번씩 방문하고 출발점으로 돌아오는 문제이다.</td>
<td>True</td>
</tr>
<tr>
<td>Subtour elimination은 TSP에서 부분 순환을 막기 위한 제약이다.</td>
<td>True</td>
</tr>
<tr>
<td>DFJ는 MTZ보다 strong하지만 제약식이 많을 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Big-M은 이진변수와 연속변수를 연결하는 데 사용될 수 있다.</td>
<td>True</td>
</tr>
</table>
---
# 8. 최종 암기 요약
이 자료는 이것만 잡으면 돼.
> **MIP**는 정수/이진 변수와 연속변수가 섞인 최적화 문제이다.
> **LP relaxation**은 정수 조건을 완화해 LP로 푸는 것이다.
> **LP relaxation 해를 반올림한다고 항상 최적해가 되는 것은 아니다.**
> **B&B = LP relaxation + Tree search**
> 좋은 formulation은 제약식 수가 적은 것이 아니라 **LP relaxation이 tight한 것**이다.
> **Knapsack**은 물건 선택 문제이다.
> **Assignment**는 일대일 배정 문제이다.
> **TSP**는 모든 도시를 한 번씩 방문하고 돌아오는 최단 경로 문제이다.
> **Subtour elimination**은 TSP에서 부분 순환을 막는 제약이다.
> **Big-M**은 이진변수와 연속변수를 연결하는 데 사용된다.

---

# RLPBL_solver

---
## 목차
---
## 1. 이 PDF의 큰 주제
이 자료는 한마디로:
> **최적화 문제를 모델링하고, Gurobi 같은 solver나 greedy heuristic으로 푸는 방법**
을 다뤄.
특히 중심 문제는 **P-median problem**이야.
---
# 2. P-median / Facility Location 개념
## 2-1. 기본 용어
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Customer / Demand point**</td>
<td>수요가 발생하는 고객 또는 지역</td>
</tr>
<tr>
<td>**Potential facility location**</td>
<td>시설을 설치할 수 있는 후보지</td>
</tr>
<tr>
<td>**Demand**</td>
<td>고객의 수요량</td>
</tr>
<tr>
<td>**Transportation cost / distance**</td>
<td>고객과 시설 사이의 거리 또는 운송비</td>
</tr>
<tr>
<td>**Decision variable**</td>
<td>시설을 열지 말지, 고객을 어느 시설에 배정할지 결정하는 변수</td>
</tr>
</table>
---
## 2-2. P-median problem
P-median은 이런 문제야.
> 후보 시설 중에서 정확히 ppp개의 시설을 선택해서,
모든 고객의 수요 × 거리 비용을 최소화하는 문제
핵심 구조:
<table header-row="true">
<tr>
<td>구분</td>
<td>내용</td>
</tr>
<tr>
<td>의사결정변수</td>
<td>어떤 시설을 선택할지, 고객을 어느 시설에 배정할지</td>
</tr>
<tr>
<td>목적</td>
<td>총 수요 × 거리 비용 최소화</td>
</tr>
<tr>
<td>제약</td>
<td>모든 고객은 하나의 열린 시설에 배정, 정확히 ppp개의 시설 선택</td>
</tr>
</table>
시험용 문장:
> **P-median problem은 후보 시설 중 p개를 선택하여 고객들의 가중 거리 비용을 최소화하는 문제이다.**
---
## 2-3. UFLP와 P-median 차이
<table header-row="true">
<tr>
<td>문제</td>
<td>핵심</td>
</tr>
<tr>
<td>**Uncapacitated Facility Location Problem**</td>
<td>시설 개설 비용 + 고객 서비스 비용 최소화</td>
</tr>
<tr>
<td>**P-median Problem**</td>
<td>정확히 ppp개의 시설을 선택하고 운송/거리 비용 최소화</td>
</tr>
</table>
쉽게 말하면:
> UFLP는 “시설을 몇 개 열지”도 비용을 보고 결정하는 느낌이고,
P-median은 “시설 개수 p개가 정해져 있고, 어디에 열지”를 결정하는 문제야.
---
# 3. P vs NP / NP-hard
자료에서 중요한 부분이야.
## 3-1. P 문제
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**P, Polynomial**</td>
<td>문제 크기가 커져도 비교적 빠르게 풀 수 있는 문제</td>
</tr>
<tr>
<td>예시</td>
<td>최단 경로 문제, 다익스트라 알고리즘</td>
</tr>
</table>
---
## 3-2. NP-hard 문제
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**NP-hard**</td>
<td>문제 규모가 커지면 경우의 수가 폭발해서 풀기 어려운 문제</td>
</tr>
<tr>
<td>예시</td>
<td>P-median, TSP, IP, MIP</td>
</tr>
</table>
P-median이 어려운 이유:
> 후보지 nnn개 중에서 ppp개를 고르는 조합 문제이기 때문
시험용 문장:
> **P-median은 후보 시설 중 p개를 선택하는 조합 문제가 되므로 NP-hard이다.**
---
# 4. 문제를 푸는 방법
자료에서는 크게 세 가지 흐름이 나와.
## 4-1. Solver
예시:
- Gurobi
- CPLEX
- Google OR-Tools
특징:
<table header-row="true">
<tr>
<td>장점</td>
<td>단점</td>
</tr>
<tr>
<td>최적해를 찾거나 보장할 수 있음</td>
<td>큰 문제에서는 시간이 오래 걸림</td>
</tr>
</table>
---
## 4-2. Heuristic
대표 예시:
- Greedy approach
특징:
<table header-row="true">
<tr>
<td>장점</td>
<td>단점</td>
</tr>
<tr>
<td>빠르게 해를 찾음</td>
<td>최적해를 보장하지 않음</td>
</tr>
</table>
---
## 4-3. Metaheuristic
예시:
- Simulated annealing
- Genetic algorithm
- Variable neighborhood search
특징:
> 하나의 해를 만든 뒤 반복적으로 개선하는 방식
---
# 5. Lower Bound / Upper Bound / Gap
이 자료에서 정말 중요한 개념이야.
## 5-1. Lower Bound, LB
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Lower Bound**</td>
<td>최소화 문제에서 최적값이 이것보다 작을 수 없다는 기준</td>
</tr>
<tr>
<td>자료 표현</td>
<td>LP relaxation의 해</td>
</tr>
<tr>
<td>특징</td>
<td>정수 조건을 완화했기 때문에 실제로 구현 불가능한 해일 수 있음</td>
</tr>
</table>
---
## 5-2. Upper Bound, UB
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Upper Bound**</td>
<td>현재까지 찾은 실현 가능한 해의 값</td>
</tr>
<tr>
<td>특징</td>
<td>정수 조건을 만족하는 feasible solution</td>
</tr>
</table>
---
## 5-3. Gap
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Gap**</td>
<td>UB와 LB의 차이</td>
</tr>
<tr>
<td>목표</td>
<td>Gap을 줄여나가는 것</td>
</tr>
<tr>
<td>Gap ≈ 0</td>
<td>현재 UB를 최적해로 볼 수 있음</td>
</tr>
</table>
시험용 문장:
> **IP/MIP solver는 Lower Bound와 Upper Bound의 gap을 줄여가며 최적해를 찾는다.**
---
# 6. Branch & Bound
자료에서는 이렇게 설명해.
> **Branch & Bound 등의 알고리즘을 사용해서 탐색 공간을 분할하며 LB는 올라가고 UB는 내려간다.**
핵심만 알면 돼.
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Branch**</td>
<td>경우를 나누어 탐색</td>
</tr>
<tr>
<td>**Bound**</td>
<td>LB/UB를 이용해 불필요한 탐색 제거</td>
</tr>
<tr>
<td>목적</td>
<td>Gap을 줄여 최적해 확인</td>
</tr>
</table>
시험용 문장:
> **Branch & Bound는 탐색 공간을 나누고 bound를 이용하여 IP/MIP 문제를 푸는 대표적인 방법이다.**
---
# 7. Greedy Approach
## 7-1. Forward Greedy Algorithm
자료에서 나온 핵심:
> 빈 집합에서 시작해서 시설을 하나씩 추가하는 방식
진행 순서:
<table header-row="true">
<tr>
<td>단계</td>
<td>내용</td>
</tr>
<tr>
<td>시작</td>
<td>아무 시설도 선택하지 않음</td>
</tr>
<tr>
<td>탐색</td>
<td>후보 시설을 하나씩 추가해봄</td>
</tr>
<tr>
<td>선택</td>
<td>총비용이 가장 낮아지는 시설 선택</td>
</tr>
<tr>
<td>반복</td>
<td>ppp개가 될 때까지 반복</td>
</tr>
</table>
---
## 7-2. Greedy의 특징
<table header-row="true">
<tr>
<td>특징</td>
<td>의미</td>
</tr>
<tr>
<td>**Construction heuristic**</td>
<td>해를 처음부터 만들어가는 방법</td>
</tr>
<tr>
<td>**Myopic approach**</td>
<td>매 단계에서 지금 당장 가장 좋아 보이는 선택</td>
</tr>
<tr>
<td>**빠름**</td>
<td>계산 시간이 짧음</td>
</tr>
<tr>
<td>**최적해 보장 X**</td>
<td>Global optimum을 보장하지 않음</td>
</tr>
</table>
시험용 문장:
> **Greedy approach는 매 단계에서 가장 좋아 보이는 선택을 하지만, 항상 전역 최적해를 보장하지는 않는다.**
---
# 8. Gurobi Solver 관련 개념
## 8-1. Gurobi의 역할
Gurobi는 최적화 문제를 풀어주는 **MIP solver**야.
이 자료에서는:
> 모델링 → Solver 입력 → 결과 해석
이 흐름이 중요해.
---
## 8-2. 변수 타입
<table header-row="true">
<tr>
<td>변수 타입</td>
<td>의미</td>
<td>예시</td>
</tr>
<tr>
<td>**CONTINUOUS**</td>
<td>연속형 변수</td>
<td>1.5, 3.14 가능</td>
</tr>
<tr>
<td>**BINARY**</td>
<td>이진 변수</td>
<td>0 또는 1</td>
</tr>
<tr>
<td>**INTEGER**</td>
<td>정수 변수</td>
<td>사람 수, 차량 대수</td>
</tr>
</table>
시험용 문장:
> **시설 설치 여부처럼 열거나 열지 않는 결정은 binary variable로 표현할 수 있다.**
---
## 8-3. Solver 결과 해석
자료에서 나온 표현:
<table header-row="true">
<tr>
<td>용어</td>
<td>의미</td>
</tr>
<tr>
<td>**Incumbent**</td>
<td>현재까지 찾은 가장 좋은 feasible solution, 즉 Upper Bound</td>
</tr>
<tr>
<td>**BestBd**</td>
<td>현재 Lower Bound</td>
</tr>
<tr>
<td>**Gap**</td>
<td>Incumbent와 BestBd의 차이</td>
</tr>
</table>
시험용 문장:
> **Incumbent는 현재까지 찾은 가장 좋은 feasible solution이고, BestBd는 lower bound이다.**
---
# 9. Large Case에서의 핵심
문제 규모가 커지면 결정변수와 제약식 수가 크게 증가해.
자료 핵심:
> Large case는 Simple case보다 결정변수와 제약식이 훨씬 많아지고, 계산 시간이 증가한다.
비교:
<table header-row="true">
<tr>
<td>방법</td>
<td>Solution quality</td>
<td>Computation time</td>
</tr>
<tr>
<td>**Gurobi / OR-Tools**</td>
<td>높음</td>
<td>김</td>
</tr>
<tr>
<td>**Greedy**</td>
<td>낮을 수 있음</td>
<td>짧음</td>
</tr>
</table>
시험용 문장:
> **Solver는 해의 품질이 높지만 시간이 오래 걸릴 수 있고, Greedy는 빠르지만 해의 품질이 낮을 수 있다.**
---
# 10. Batch Run
## 10-1. 왜 필요한가?
자료에서는 하나의 random seed만으로 실험하면 신뢰성이 부족하다고 말해.
즉:
> 고객 위치, 시설 후보지, 수요가 random으로 생성되기 때문에 여러 seed로 반복 실험해야 신뢰성이 있다.
## 10-2. Batch run의 의미
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Batch run**</td>
<td>여러 조건이나 여러 random seed에 대해 반복 실험하는 것</td>
</tr>
<tr>
<td>목적</td>
<td>결과의 신뢰성 확인, ppp 변화에 따른 cost 변화 분석</td>
</tr>
</table>
시험용 문장:
> **하나의 random seed 결과만으로는 신뢰성이 부족하므로 batch run을 통해 여러 난수에 대해 실험할 수 있다.**
---
# 11. Deterministic Inventory Model
후반부는 재고모형을 다뤄.
## 11-1. Deterministic Inventory Model이란?
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Deterministic**</td>
<td>수요 등 파라미터가 이미 알려져 있음</td>
</tr>
<tr>
<td>**Inventory model**</td>
<td>주문량, 재고, 비용을 결정하는 모델</td>
</tr>
</table>
---
## 11-2. Sequence of Events
매 기간 ttt마다 이벤트가 다음 순서로 발생해.
<table header-row="true">
<tr>
<td>순서</td>
<td>내용</td>
</tr>
<tr>
<td>1</td>
<td>보충 주문, 주문하면 즉시 입고</td>
</tr>
<tr>
<td>2</td>
<td>수요 발생, 재고에서 충족</td>
</tr>
<tr>
<td>3</td>
<td>비용 산정, 기간 말 재고 기준으로 유지비 발생</td>
</tr>
</table>
시험용 문장:
> **이 deterministic inventory model에서는 주문이 발생하면 즉시 입고되고, 이후 수요가 발생하며, 기간 말 재고 기준으로 비용이 계산된다.**
---
## 11-3. 주요 의사결정변수
<table header-row="true">
<tr>
<td>변수</td>
<td>의미</td>
</tr>
<tr>
<td>qtq_tqt</td>
<td>시점 ttt에 주문하는 수량</td>
</tr>
<tr>
<td>yty_tyt</td>
<td>시점 ttt에 주문하는지 여부</td>
</tr>
<tr>
<td>xtx_txt</td>
<td>시점 ttt 종료 시 재고 수준</td>
</tr>
</table>
핵심:
> yty_tyt는 주문 여부이므로 binary variable로 볼 수 있다.
---
## 11-4. Setup Cost / Holding Cost
<table header-row="true">
<tr>
<td>비용</td>
<td>의미</td>
</tr>
<tr>
<td>**Setup cost**</td>
<td>주문 또는 생산을 시작할 때 발생하는 고정비</td>
</tr>
<tr>
<td>**Holding cost**</td>
<td>남은 재고를 보유하는 데 드는 비용</td>
</tr>
</table>
---
# 12. Stockout / Lost Sales
자료에서는 lost sales penalty를 다뤄.
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Stockout**</td>
<td>수요를 재고로 충족하지 못한 상황</td>
</tr>
<tr>
<td>**Lost sales**</td>
<td>충족하지 못하고 잃어버린 판매</td>
</tr>
<tr>
<td>**Penalty cost ppp**</td>
<td>미충족 수요 1단위당 비용</td>
</tr>
<tr>
<td>**ztz_tzt**</td>
<td>시점 ttt의 미충족 수요량</td>
</tr>
</table>
시험용 문장:
> **Lost sales penalty가 커질수록 미충족 수요를 줄이는 방향의 주문 정책이 선호될 수 있다.**
---
# 13. Lead Time / Pipeline Inventory
## 13-1. Lead Time
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Lead time LLL**</td>
<td>주문 후 실제 도착까지 걸리는 기간</td>
</tr>
</table>
즉:
> 주문했다고 바로 들어오는 것이 아니라 LLL기간 후 도착할 수 있다.
---
## 13-2. Pipeline Inventory
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Pipeline inventory**</td>
<td>이미 주문했지만 아직 도착하지 않은 재고</td>
</tr>
<tr>
<td>oto_tot</td>
<td>시점 ttt의 pipeline inventory</td>
</tr>
<tr>
<td>NtN_tNt</td>
<td>계획 기간 이전에 주문되어 ttt에 도착 예정인 물량</td>
</tr>
</table>
시험용 문장:
> **Pipeline inventory는 발주되었지만 아직 도착하지 않은 재고를 의미한다.**
---
# 14. 이 자료에서 안 외워도 되는 것
시험이 quiz 형식이면 아래는 낮은 우선순위야.
- Gurobi license 발급 절차 세부 단계
- 코드 스크린샷 전체
- 특정 seed의 결과 숫자
- P-median 예제의 모든 거리표 숫자
- Gurobi 설치 링크
- Homework 제출 형식
하지만 **개념 이름과 역할**은 알아두는 게 좋아.
---
# 15. True/False 대비
<table header-row="true">

<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>P-median은 후보 시설 중 p개를 선택하는 문제이다.</td>
<td>True</td>
</tr>
<tr>
<td>P-median의 목적은 고객의 가중 거리 비용을 최소화하는 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>P-median은 항상 쉽게 풀리는 P 문제이다.</td>
<td>False</td>
</tr>
<tr>
<td>NP-hard 문제는 규모가 커질수록 계산량이 폭발할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Greedy는 항상 전역 최적해를 보장한다.</td>
<td>False</td>
</tr>
<tr>
<td>Greedy는 매 단계에서 지금 가장 좋아 보이는 선택을 한다.</td>
<td>True</td>
</tr>
<tr>
<td>Solver는 최적해를 보장할 수 있지만 시간이 오래 걸릴 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Lower Bound는 정수 조건을 완화한 LP relaxation 해와 관련될 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Upper Bound는 feasible solution에서 나온 값이다.</td>
<td>True</td>
</tr>
<tr>
<td>Gap이 0에 가까워지면 최적해에 가까워진다.</td>
<td>True</td>
</tr>
<tr>
<td>Incumbent는 현재까지 찾은 가장 좋은 feasible solution이다.</td>
<td>True</td>
</tr>
<tr>
<td>BestBd는 lower bound를 의미한다.</td>
<td>True</td>
</tr>
<tr>
<td>하나의 random seed 결과만으로 항상 충분히 신뢰할 수 있다.</td>
<td>False</td>
</tr>
<tr>
<td>Batch run은 여러 조건 또는 seed에 대해 반복 실험하는 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>Deterministic inventory model에서는 수요가 주어진 값으로 다뤄진다.</td>
<td>True</td>
</tr>
<tr>
<td>Pipeline inventory는 주문했지만 아직 도착하지 않은 재고이다.</td>
<td>True</td>
</tr>
</table>
---
# 최종 암기 요약
이 자료는 아래 문장들만 확실히 잡으면 돼.
> **P-median은 후보 시설 중 p개를 선택하여 고객의 가중 거리 비용을 최소화하는 문제이다.**
> **NP-hard 문제는 규모가 커지면 경우의 수가 폭발하여 풀기 어렵다.**
> **Greedy는 빠르지만 전역 최적해를 보장하지 않는 heuristic이다.**
> **Solver는 해의 품질이 높지만 계산 시간이 길어질 수 있다.**
> **IP/MIP solver는 LB와 UB의 gap을 줄여가며 최적해를 찾는다.**
> **Incumbent는 현재까지 찾은 가장 좋은 feasible solution이고, BestBd는 lower bound이다.**
> **Batch run은 여러 random seed나 조건에서 반복 실험하여 결과의 신뢰성을 확인하는 것이다.**
> **Deterministic inventory model은 수요와 파라미터가 주어진 상태에서 주문량, 주문 여부, 재고 수준을 결정하는 모델이다.**
> **Lead time은 주문 후 도착까지 걸리는 시간이고, pipeline inventory는 주문했지만 아직 도착하지 않은 재고이다.**

---

# RLPBL_Simopt

---
## 목차
---
## 1. 이 PDF의 큰 주제
한마디로 말하면:
> **수요가 확률적으로 발생하는 재고 문제에서, 어떤 주문 정책을 쓸지 정하고 simulation으로 좋은 파라미터를 찾는 방법**
을 다루는 자료야.
즉, 여기서는 “정확한 최적해를 수식으로 바로 구하기”보다는,
> 정책 형태를 정함
여러 파라미터를 넣어봄
Monte Carlo simulation으로 평균 비용을 계산함
가장 비용이 낮은 파라미터를 선택함
이 흐름이 중요해.
---
# 2. Stochastic Inventory Model
## 2-1. Stochastic inventory model이란?
**수요가 확률적으로 발생하는 재고모형**이야.
Deterministic inventory model은 수요가 이미 정해져 있지만,
Stochastic inventory model은 수요가 매번 달라질 수 있어.
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>**Deterministic inventory model**</td>
<td>수요가 주어진 값으로 고정됨</td>
</tr>
<tr>
<td>**Stochastic inventory model**</td>
<td>수요가 확률적으로 발생함</td>
</tr>
</table>
시험용 문장:
> **Stochastic inventory model은 수요가 확률적으로 발생하는 재고모형이다.**
---
# 3. Inventory Policy
## 3-1. Policy란?
자료에서 policy는 이렇게 설명돼.
> **재고 문제에 대한 해를 제공하는 규칙**
쉽게 말하면:
> 언제 주문할지, 얼마나 주문할지를 정하는 규칙
예를 들어:
- 매 기간마다 20개 주문
- 재고가 10개 이하가 되면 50개 주문
- 매 기간 재고를 80개까지 채움
이런 것들이 전부 policy야.
---
# 4. 대표 Inventory Policy
## 4-1. Every T years, order Q
<table header-row="true">
<tr>
<td>내용</td>
<td>의미</td>
</tr>
<tr>
<td>주문 시점</td>
<td>매 TTT 기간마다</td>
</tr>
<tr>
<td>주문량</td>
<td>항상 QQQ만큼</td>
</tr>
<tr>
<td>특징</td>
<td>단순한 정책</td>
</tr>
</table>
핵심:
> 일정한 시간 간격마다 고정 수량을 주문하는 정책
---
## 4-2. (r,Q)Policy
자료 표현:
> **When reaching r, order Q**
뜻:
> 재고가 rrr 수준에 도달하면 QQQ만큼 주문하는 정책
<table header-row="true">
<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>rrr</td>
<td>재주문점</td>
</tr>
<tr>
<td>QQQ</td>
<td>주문량</td>
</tr>
</table>
시험용 문장:
> **(r,Q)(r,Q)(r,Q) policy는 재고가 rrr에 도달하면 QQQ만큼 주문하는 정책이다.**
자료에서는 **continuous review + fixed cost** 상황에서 (r,Q)(r,Q)(r,Q) policy가 나온다고 정리돼.
---
## 4-3. Base-stock Policy (S−1,S)(S-1, S)(S−1,S)
자료 표현:
> **Every T years, order up to S**
뜻:
> 매 기간 재고를 확인하고, 재고 수준을 SSS까지 채우는 정책
<table header-row="true">
<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>SSS</td>
<td>목표 재고 수준</td>
</tr>
<tr>
<td>order up to SSS</td>
<td>재고가 SSS가 되도록 주문</td>
</tr>
</table>
예를 들어 현재 재고가 30이고 S=80S=80S=80이면 50개 주문.
시험용 문장:
> **Base-stock policy는 매 review 시점마다 재고를 목표 수준 SSS까지 채우는 정책이다.**
자료에서는 **periodic review + fixed cost 없음** 상황에서 base-stock policy가 나온다고 정리돼.
---
## 4-4. (s,S)(s, S)(s,S) Policy
자료 표현:
> **When reaching s, order up to S**
뜻:
> 재고가 sss 이하로 내려가면, 재고를 SSS까지 채우는 정책
<table header-row="true">
<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>sss</td>
<td>주문을 시작하는 기준점</td>
</tr>
<tr>
<td>SSS</td>
<td>주문 후 목표 재고 수준</td>
</tr>
</table>
시험용 문장:
> **(s,S)(s,S)(s,S) policy는 재고가 sss에 도달하면 SSS까지 주문하는 정책이다.**
자료에서는 **periodic review + fixed cost 있음** 상황에서 (s,S)(s,S)(s,S) policy가 나온다고 정리돼.
---
# 5. Policy Optimality
## 5-1. Optimal Policy란?
Optimal policy는:
> 다른 어떤 policy도 이 policy보다 더 좋은 성능을 낼 수 없는 정책
자료에서는 optimal policy를 구하는 과정을 두 단계로 설명해.
## 5-2. Optimal policy를 구하는 법
<table header-row="true">
<tr>
<td>단계</td>
<td>의미</td>
<td>예시</td>
</tr>
<tr>
<td>1단계</td>
<td>최적 policy의 형태를 찾음</td>
<td>(s,S)(s,S)(s,S) policy가 최적 형태인지 확인</td>
</tr>
<tr>
<td>2단계</td>
<td>그 policy의 최적 파라미터를 찾음</td>
<td>s∗,S∗s\^\*, S\^\*s∗,S∗ 찾기</td>
</tr>
</table>
중요한 점:
> 모든 문제에서 최적 policy의 형태를 알 수 있는 것은 아니다.
그래서 실제로는 그럴듯한 policy를 선택하고,
simulation으로 좋은 파라미터를 찾는 경우가 많아.
시험용 문장:
> **Optimal policy의 형태를 항상 알 수 있는 것은 아니므로, plausible한 policy를 선택하고 파라미터를 heuristic하게 찾기도 한다.**
---
# 6. Review 방식 구분
자료에 나온 구분이야.
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
<td>대표 정책</td>
</tr>
<tr>
<td>**Continuous review**</td>
<td>재고를 계속 관찰</td>
<td>(r,Q)(r,Q)(r,Q) policy</td>
</tr>
<tr>
<td>**Periodic review**</td>
<td>일정 기간마다 재고 확인</td>
<td>Base-stock, (s,S)(s,S)(s,S)</td>
</tr>
</table>
Fixed cost 여부도 중요해.
<table header-row="true">
<tr>
<td>상황</td>
<td>대표 정책</td>
</tr>
<tr>
<td>Fixed cost 있음</td>
<td>(r,Q)(r,Q)(r,Q), (s,S)(s,S)(s,S)</td>
</tr>
<tr>
<td>Fixed cost 없음</td>
<td>Base-stock policy</td>
</tr>
</table>
---
# 7. Inventory Process
## 7-1. Single-period / Finite horizon / Infinite horizon
<table header-row="true">
<tr>
<td>모델</td>
<td>의미</td>
</tr>
<tr>
<td>**Single-period model**</td>
<td>한 번의 판매 시즌만 고려</td>
</tr>
<tr>
<td>**Finite horizon model**</td>
<td>정해진 기간까지만 고려</td>
</tr>
<tr>
<td>**Infinite horizon model**</td>
<td>무한 기간 동안 반복되는 문제</td>
</tr>
</table>
Single-period 예시:
- 신문 판매
- 의류산업
- 짧은 판매 시즌
- 시즌이 끝나면 가치가 급격히 감소
시험용 문장:
> **Single-period inventory model은 짧은 판매 시즌처럼 한 번의 주문과 판매만 고려하는 모델이다.**
---
## 7-2. Sequence of Events
매 기간 ttt마다 사건이 이 순서로 발생해.
<table header-row="true">
<tr>
<td>순서</td>
<td>내용</td>
</tr>
<tr>
<td>1</td>
<td>Inventory level 확인</td>
</tr>
<tr>
<td>2</td>
<td>주문하고, 주문 즉시 입고</td>
</tr>
<tr>
<td>3</td>
<td>고객 수요 발생</td>
</tr>
<tr>
<td>4</td>
<td>남은 재고가 있으면 holding cost, 부족하면 stockout cost 발생</td>
</tr>
</table>
시험용 문장:
> **이 재고모형에서는 주문 후 즉시 입고되고, 그다음 수요가 발생하며, 기간 말 재고 또는 부족분에 따라 비용이 발생한다.**
---
# 8. 비용 개념
## 8-1. Holding Cost
**재고가 남았을 때 발생하는 비용**이야.
예:
> 너무 많이 주문해서 재고가 남으면 보관비가 발생함
시험용 문장:
> **Holding cost는 남은 재고를 보유하는 데 드는 비용이다.**
---
## 8-2. Stockout Cost / Underage Cost
**재고가 부족해서 수요를 만족하지 못했을 때 발생하는 비용**이야.
예:
> 고객 수요가 50인데 재고가 30이면 20개 부족
이 부족분에 대해 stockout cost 발생
시험용 문장:
> **Stockout cost는 수요를 충족하지 못했을 때 발생하는 비용이다.**
---
## 8-3. Overage Cost / Underage Cost
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Overage cost**</td>
<td>너무 많이 주문해서 남는 비용</td>
</tr>
<tr>
<td>**Underage cost**</td>
<td>너무 적게 주문해서 부족한 비용</td>
</tr>
</table>
---
# 9. Finite Horizon Inventory Problem
자료에서는 T=100T=100T=100인 finite horizon inventory problem을 다뤄.
## 9-1. 주요 파라미터
<table header-row="true">
<tr>
<td>파라미터</td>
<td>의미</td>
</tr>
<tr>
<td>초기 재고</td>
<td>처음 시작할 때의 재고</td>
</tr>
<tr>
<td>Setup cost KKK</td>
<td>주문할 때 발생하는 고정비</td>
</tr>
<tr>
<td>Capacity CCC</td>
<td>창고 최대 재고 용량</td>
</tr>
<tr>
<td>최대 주문량 MMM</td>
<td>한 기간에 주문 가능한 최대량</td>
</tr>
<tr>
<td>단위 주문 비용 ccc</td>
<td>한 단위 주문 비용</td>
</tr>
<tr>
<td>결품 비용 ppp</td>
<td>부족분에 대한 비용</td>
</tr>
<tr>
<td>재고 유지 비용 hhh</td>
<td>남은 재고 보유 비용</td>
</tr>
</table>
## 9-2. 주요 의사결정변수
<table header-row="true">
<tr>
<td>변수</td>
<td>의미</td>
</tr>
<tr>
<td>주문량</td>
<td>각 기간에 얼마나 주문할지</td>
</tr>
<tr>
<td>주문 여부</td>
<td>주문하면 1, 안 하면 0</td>
</tr>
<tr>
<td>재고 수준</td>
<td>기간 말 재고</td>
</tr>
<tr>
<td>부족량</td>
<td>수요를 만족하지 못한 양</td>
</tr>
</table>
핵심:
> 수요가 확률적으로 발생하기 때문에, 한 번의 계산으로 정확한 최적 정책을 찾기 어렵다.
---
# 10. Simulation-based Optimization
## 10-1. 의미
Simulation-based optimization은:
> simulation을 반복해서 실행하면서 가장 좋은 policy parameter를 찾는 방법
자료에서는 Monte Carlo simulation을 사용해.
## 10-2. Monte Carlo Simulation
Monte Carlo simulation은:
> 수요 시나리오를 여러 번 랜덤으로 생성해서 평균 비용을 계산하는 방식
예를 들어 S=30S=30S=30, S=40S=40S=40, S=50S=50S=50을 각각 넣고,
각각 1000번씩 시뮬레이션해서 평균 비용이 가장 낮은 SSS를 고르는 식이야.
시험용 문장:
> **Simulation-based optimization은 여러 수요 시나리오를 생성해 policy의 평균 비용을 평가하고, 가장 좋은 파라미터를 선택하는 방법이다.**
---
# 11. Base-stock Policy의 파라미터 탐색
Base-stock policy에서는 SSS 하나를 찾으면 돼.
흐름:
<table header-row="true">
<tr>
<td>단계</td>
<td>내용</td>
</tr>
<tr>
<td>1</td>
<td>여러 수요 시나리오 생성</td>
</tr>
<tr>
<td>2</td>
<td>후보 SSS 값을 하나씩 넣어 simulation</td>
</tr>
<tr>
<td>3</td>
<td>각 SSS의 평균 비용 계산</td>
</tr>
<tr>
<td>4</td>
<td>평균 비용이 가장 낮은 S∗S\^\*S∗ 선택</td>
</tr>
</table>
시험용 문장:
> **Base-stock policy에서는 simulation을 통해 평균 비용이 가장 낮은 S∗S\^\*S∗를 찾을 수 있다.**
---
# 12. (s,S)(s,S)(s,S) Policy의 파라미터 탐색
(s,S)(s,S)(s,S) policy에서는 sss와 SSS 두 개를 찾아야 해.
흐름:
<table header-row="true">
<tr>
<td>단계</td>
<td>내용</td>
</tr>
<tr>
<td>1</td>
<td>여러 수요 시나리오 생성</td>
</tr>
<tr>
<td>2</td>
<td>가능한 s,Ss, Ss,S 조합을 탐색</td>
</tr>
<tr>
<td>3</td>
<td>각 조합의 평균 비용 계산</td>
</tr>
<tr>
<td>4</td>
<td>평균 비용이 가장 낮은 (s∗,S∗)(s\^\*, S\^\*)(s∗,S∗) 선택</td>
</tr>
</table>
시험용 문장:
> **(s,S)(s,S)(s,S) policy에서는 simulation을 통해 평균 비용이 가장 낮은 (s∗,S∗)(s\^\*,S\^\*)(s∗,S∗)를 찾을 수 있다.**
---
# 13. Lower Bound / Wait-and-See Solution
## 13-1. 왜 필요한가?
최적 s∗,S∗s\^\*, S\^\*s∗,S∗를 찾았다고 해도,
그 policy가 얼마나 좋은지 평가해야 해.
그래서 비교 기준으로 **Lower Bound**를 사용해.
---
## 13-2. Wait-and-See Solution이란?
자료에서 핵심은 이거야.
> 만약 실현될 수요 정보를 미리 다 알고 있다면?
이게 Wait-and-See solution이야.
즉:
> 미래 수요를 미리 알고 있는 Oracle이 푼 해
현실에서는 미래 수요를 미리 알 수 없기 때문에 실제로 구현 가능한 정책은 아니야.
하지만 수요를 미리 알고 푼 해는 매우 유리하므로,
실제 policy가 이보다 더 좋기는 어려워.
그래서 **lower bound**로 사용돼.
시험용 문장:
> **Wait-and-See solution은 미래 수요를 미리 알고 있다고 가정한 비현실적인 해이며, stochastic inventory problem에서 lower bound로 사용될 수 있다.**
---
# 14. Lower Bound / Upper Bound / Gap
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Lower Bound, LB**</td>
<td>실제 가능한 정책이 이보다 더 좋기 어려운 기준값</td>
</tr>
<tr>
<td>**Policy cost**</td>
<td>우리가 선택한 policy의 simulation 평균 비용</td>
</tr>
<tr>
<td>**Gap**</td>
<td>policy cost가 lower bound와 얼마나 차이 나는지 보는 지표</td>
</tr>
</table>
자료에서는 Wait-and-See solution을 lower bound로 보고,
policy 성능을 이 lower bound와 비교해.
시험용 문장:
> **Policy의 품질은 Wait-and-See lower bound와의 gap을 통해 평가할 수 있다.**
---
# 15. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 우선순위 낮아.
- 알고리즘 pseudo-code 전체
- Python 파일명
- simulation 결과 그래프의 세부 숫자
- T=100T=100T=100 같은 특정 파라미터 숫자 전체
- 코드 구현 세부 내용
- Homework 세부 제출 내용
하지만 아래는 알아야 해.
- policy 종류
- (r,Q)(r,Q)(r,Q), base-stock, (s,S) 의미
(s,S)(s,S)
- simulation-based optimization
- Monte Carlo simulation
- Wait-and-See lower bound
- sequence of events
- holding cost, stockout cost
---
# 16. True/False 대비
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>Stochastic inventory model에서는 수요가 확률적으로 발생한다.</td>
<td>True</td>
</tr>
<tr>
<td>Policy는 언제 얼마나 주문할지 정하는 규칙이다.</td>
<td>True</td>
</tr>
<tr>
<td>(r,Q)(r,Q)(r,Q) policy는 재고가 rrr에 도달하면 QQQ만큼 주문한다.</td>
<td>True</td>
</tr>
<tr>
<td>Base-stock policy는 재고를 목표 수준 SSS까지 채우는 정책이다.</td>
<td>True</td>
</tr>
<tr>
<td>(s,S)(s,S)(s,S) policy는 재고가 sss에 도달하면 SSS까지 주문한다.</td>
<td>True</td>
</tr>
<tr>
<td>Optimal policy의 형태는 모든 문제에서 항상 알려져 있다.</td>
<td>False</td>
</tr>
<tr>
<td>Monte Carlo simulation은 여러 랜덤 수요 시나리오를 사용해 평균 성능을 평가한다.</td>
<td>True</td>
</tr>
<tr>
<td>Simulation-based optimization은 policy parameter를 찾는 데 사용할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Holding cost는 재고가 부족할 때 발생하는 비용이다.</td>
<td>False</td>
</tr>
<tr>
<td>Stockout cost는 수요를 만족하지 못할 때 발생하는 비용이다.</td>
<td>True</td>
</tr>
<tr>
<td>Wait-and-See solution은 미래 수요를 미리 안다고 가정한다.</td>
<td>True</td>
</tr>
<tr>
<td>Wait-and-See solution은 현실에서 그대로 구현 가능한 정책이다.</td>
<td>False</td>
</tr>
<tr>
<td>Wait-and-See solution은 lower bound로 사용할 수 있다.</td>
<td>True</td>
</tr>
</table>
---
# 최종 암기 요약
이 자료는 아래만 확실히 잡으면 돼.
> **Policy는 재고 문제에서 언제, 얼마나 주문할지를 정하는 규칙이다.**
> **(r,Q)(r,Q)(r,Q) policy는 재고가 rrr에 도달하면 QQQ만큼 주문한다.**
> **Base-stock policy는 매 review 시점마다 재고를 SSS까지 채운다.**
> **(s,S)(s,S)(s,S) policy는 재고가 sss에 도달하면 SSS까지 주문한다.**
> **Optimal policy는 다른 어떤 policy도 더 좋은 성능을 낼 수 없는 정책이다.**
> **최적 policy 형태를 모르는 경우, 그럴듯한 policy를 선택하고 simulation으로 파라미터를 찾는다.**
> **Simulation-based optimization은 Monte Carlo simulation으로 평균 비용을 계산해 좋은 policy parameter를 찾는 방법이다.**
> **Wait-and-See solution은 미래 수요를 미리 알고 푼 비현실적인 해이며 lower bound로 사용된다.**
> **Policy 성능은 Wait-and-See lower bound와의 gap으로 평가할 수 있다.**

---

# 06] Markov Decision Processes

---
## 목차
---
## 1. 이 PDF의 큰 주제
한마디로 말하면:
> **강화학습 문제를 수학적으로 표현하는 기본 틀인 MDP를 배우는 자료**
야.
MDP는 **stochastic dynamic environment**, 즉 확률적으로 변하는 동적 환경에서
agent가 순차적으로 의사결정을 하는 문제를 표현해.
핵심 문장:
> **MDP는 강화학습의 underlying environment이다.**
즉, 강화학습 알고리즘을 배우기 전에
먼저 state, action, reward, transition, policy, value function을 이해해야 해.
---
# 2. MDP의 목표
MDP의 목표는:
> **최적 정책을 찾는 것**
자료 표현으로는:
> Goal: finding optimal policy π(a∣s)\\pi(a\|s)π(a∣s)
여기서 policy는 상태 sss에서 행동 aaa를 선택할 확률이야.
그리고 최적 정책을 찾기 위해서는:
> **state value function vπ(s)v_\\pi(s)vπ(s)** 를 추정해야 해.
---
# 3. Agent-Environment Interface
## 기본 구조
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Agent**</td>
<td>학습하고 의사결정을 하는 주체</td>
</tr>
<tr>
<td>**Environment**</td>
<td>agent가 상호작용하는 대상</td>
</tr>
<tr>
<td>**State**</td>
<td>현재 상황</td>
</tr>
<tr>
<td>**Action**</td>
<td>agent가 선택하는 행동</td>
</tr>
<tr>
<td>**Reward**</td>
<td>행동 결과로 받는 보상</td>
</tr>
</table>
흐름은 이렇게 돼.
> Agent가 action을 선택한다.
Environment가 그 action에 반응한다.
Environment는 reward와 new state를 agent에게 준다.
시험용 문장:
> **Agent는 action을 선택하고, environment는 reward와 next state를 제공한다.**
---
# 4. Interaction Sequence
매 시점 ttt마다 다음 순서로 진행돼.
<table header-row="true">
<tr>
<td>순서</td>
<td>내용</td>
</tr>
<tr>
<td>1</td>
<td>Agent가 현재 상태 StS_tSt를 관찰</td>
</tr>
<tr>
<td>2</td>
<td>Agent가 행동 AtA_tAt 선택</td>
</tr>
<tr>
<td>3</td>
<td>한 단계 후 reward Rt+1R_\{t+1\}Rt+1를 받음</td>
</tr>
<tr>
<td>4</td>
<td>새로운 상태 St+1S_\{t+1\}St+1로 이동</td>
</tr>
</table>
MDP의 trajectory는 이런 순서야.
S0,A0,R1,S1,A1,R2,S2,⋯S_0, A_0, R_1, S_1, A_1, R_2, S_2, \\cdots
S0,A0,R1,S1,A1,R2,S2,⋯
시험 포인트:
> reward는 RtR_tRt가 아니라, 행동 후 받는 Rt+1R_\{t+1\}Rt+1로 표현된다.
---
# 5. Finite MDP
Finite MDP는:
> state, action, reward의 집합이 모두 유한한 MDP
즉,
<table header-row="true">
<tr>
<td>구성요소</td>
<td>의미</td>
</tr>
<tr>
<td>SSS</td>
<td>가능한 상태들의 집합</td>
</tr>
<tr>
<td>AAA</td>
<td>가능한 행동들의 집합</td>
</tr>
<tr>
<td>RRR</td>
<td>가능한 보상들의 집합</td>
</tr>
</table>
자료에서 중요한 함수는 이거야.
p(s′,r∣s,a)p(s', r \| s, a)
p(s′,r∣s,a)
뜻:
> 현재 상태 sss에서 행동 aaa를 했을 때,
다음 상태가 s′s's′가 되고 reward가 rrr일 확률
시험용 문장:
> **p(s′,r∣s,a)p(s', r\|s,a)p(s′,r∣s,a)는 MDP의 dynamics를 정의한다.**
---
# 6. Markov Property
이 PDF에서 매우 중요한 개념이야.
## Markov property란?
> 미래는 과거 전체가 아니라 **현재 상태와 현재 행동에만 의존한다**는 성질
즉, 다음 상태와 보상은
지금까지의 모든 history를 다 볼 필요 없이
> 현재 상태 St−1S_\{t-1\}St−1와 현재 행동 At−1A_\{t-1\}At−1
만 알면 결정된다고 보는 거야.
시험용 문장:
> **Markov property는 다음 상태와 reward가 과거 전체가 아니라 직전 state와 action에만 의존한다는 성질이다.**
예상 문제:
> MDP에서는 다음 상태가 항상 전체 과거 history에 의존한다.
→ **False**
---
# 7. State-transition과 Reward
자료에서는 p(s′,r∣s,a)p(s',r\|s,a)p(s′,r∣s,a)에서 여러 정보를 계산할 수 있다고 해.
## State-transition probability
p(s′∣s,a)p(s'\|s,a)
p(s′∣s,a)
뜻:
> 상태 sss에서 행동 aaa를 했을 때 다음 상태가 s′s's′가 될 확률
---
## Expected reward
r(s,a)r(s,a)
r(s,a)
뜻:
> 상태 sss에서 행동 aaa를 했을 때 기대되는 평균 reward
quiz에서는 수식을 외우기보다 이 의미만 알면 돼.
---
# 8. MDP 구성요소
자료에서 말하는 MDP 구성요소는 이렇게 정리하면 돼.
<table header-row="true">
<tr>
<td>구성요소</td>
<td>의미</td>
</tr>
<tr>
<td>**Time step**</td>
<td>의사결정이 이루어지는 단계</td>
</tr>
<tr>
<td>**Action**</td>
<td>agent가 선택할 수 있는 행동</td>
</tr>
<tr>
<td>**State**</td>
<td>현재 상황을 나타내는 정보</td>
</tr>
<tr>
<td>**Reward**</td>
<td>agent의 목표를 정의하는 신호</td>
</tr>
</table>
중요한 점:
> Time step은 실제 시간 간격일 필요는 없고, 의사결정 단계일 수 있다.
---
# 9. Recycling Robot 예제
이 예제는 MDP 구성요소를 보여주는 예시야.
<table header-row="true">
<tr>
<td>구성요소</td>
<td>내용</td>
</tr>
<tr>
<td>State</td>
<td>배터리 상태: high, low</td>
</tr>
<tr>
<td>Action</td>
<td>search, wait, recharge</td>
</tr>
<tr>
<td>Reward</td>
<td>행동 결과에 따른 보상</td>
</tr>
<tr>
<td>Transition</td>
<td>행동에 따라 high/low 상태가 확률적으로 변함</td>
</tr>
</table>
핵심:
> 같은 action을 해도 다음 상태가 확률적으로 달라질 수 있다.
예를 들어 배터리가 high일 때 search를 하면
계속 high일 수도 있고 low로 떨어질 수도 있어.
---
# 10. Goals and Rewards
강화학습의 목표는:
> **장기적인 누적 reward를 최대화하는 것**
중요한 점은 **즉각적인 reward만 최대화하는 것이 아니라 cumulative reward를 최대화**한다는 거야.
자료 예시:
<table header-row="true">
<tr>
<td>문제</td>
<td>reward 예시</td>
</tr>
<tr>
<td>미로 탈출</td>
<td>탈출 전까지 -1</td>
</tr>
<tr>
<td>체스</td>
<td>승리 +1, 패배 -1, 무승부 0</td>
</tr>
</table>
중요 문장:
> **Reward는 agent가 우리가 원하는 목표를 달성하도록 설계해야 한다.**
주의점:
> sub-goal에 reward를 잘못 주면 agent가 진짜 목표를 놓칠 수 있다.
---
# 11. Return과 Episode
## Return GtG_tGt
Return은:
> 현재 시점 이후 받을 reward들의 합
가장 단순한 형태는:
Gt=Rt+1+Rt+2+⋯+RTG_t = R_\{t+1\} + R_\{t+2\} + \\cdots + R_T
Gt=Rt+1+Rt+2+⋯+RT
즉, reward 하나가 아니라 **앞으로 받을 전체 reward**야.
---
## Episode
Episode는:
> 시작부터 terminal state에 도달할 때까지의 하나의 실행 단위
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Terminal state**</td>
<td>episode가 끝나는 상태</td>
</tr>
<tr>
<td>**Episodic task**</td>
<td>episode 단위로 끝나는 문제</td>
</tr>
</table>
예시:
- 게임 한 판
- 미로 탈출 한 번
- 체스 한 게임
시험용 문장:
> **Episodic task는 terminal state에서 episode가 끝나는 문제이다.**
---
# 12. Discounting
계속 진행되는 문제에서는 reward를 무한히 더하면 값이 무한대가 될 수 있어.
그래서 discount rate γ\\gammaγ를 사용해.
Gt=Rt+1+γRt+2+γ2Rt+3+⋯G_t = R_\{t+1\} + \\gamma R_\{t+2\} + \\gamma\^2 R_\{t+3\} + \\cdots
Gt=Rt+1+γRt+2+γ2Rt+3+⋯
## γ\\gammaγ의 의미
<table header-row="true">
<tr>
<td>γ\\gammaγ 값</td>
<td>의미</td>
</tr>
<tr>
<td>γ→0\\gamma \\to 0γ→0</td>
<td>당장의 reward를 중요하게 봄, myopic</td>
</tr>
<tr>
<td>γ→1\\gamma \\to 1γ→1</td>
<td>미래 reward도 중요하게 봄, farsighted</td>
</tr>
</table>
시험용 문장:
> **Discount rate γ\\gammaγ는 미래 reward의 현재 가치를 결정한다.**
---
# 13. Policy
Policy는:
> state에서 action을 선택하는 규칙
자료 표현:
π(a∣s)\\pi(a\|s)
π(a∣s)
뜻:
> 상태 sss에서 행동 aaa를 선택할 확률
즉, policy는 각 state마다 가능한 action에 대한 확률분포야.
시험용 문장:
> **Policy π(a∣s)\\pi(a\|s)π(a∣s)는 상태 sss에서 행동 aaa를 선택할 확률이다.**
---
# 14. Value Function
## State-value function vπ(s)v_\\pi(s)vπ(s)
뜻:
> policy π\\piπ를 따를 때, 상태 sss가 장기적으로 얼마나 좋은지를 나타내는 함수
쉽게 말하면:
> 이 상태에서 시작하면 앞으로 reward를 얼마나 받을 것 같은가?
---
## Action-value function qπ(s,a)q_\\pi(s,a)qπ(s,a)
뜻:
> policy π\\piπ를 따를 때, 상태 sss에서 행동 aaa를 했을 때의 장기적 가치
쉽게 말하면:
> 이 상태에서 이 행동을 하면 앞으로 reward를 얼마나 받을 것 같은가?
---
## vvv와 qqq 차이
<table header-row="true">
<tr>
<td>개념</td>
<td>기준</td>
</tr>
<tr>
<td>vπ(s)v_\\pi(s)vπ(s)</td>
<td>상태 sss의 가치</td>
</tr>
<tr>
<td>qπ(s,a)q_\\pi(s,a)qπ(s,a)</td>
<td>상태 sss에서 행동 aaa를 했을 때의 가치</td>
</tr>
</table>
시험용 문장:
> **vπ(s)v_\\pi(s)vπ(s)는 state-value function이고, qπ(s,a)q_\\pi(s,a)qπ(s,a)는 action-value function이다.**
---
# 15. Bellman Equation
Bellman equation은 이 자료에서 핵심 중 핵심이야.
## 의미
Bellman equation은:
> 현재 state의 value를
즉시 reward + 다음 state의 value로 표현하는 관계식
즉,
> 현재 가치 = 지금 받을 reward + 미래 가치
자료 표현으로는:
> state의 value와 successor states의 value 사이의 관계
시험용 문장:
> **Bellman equation은 한 상태의 가치와 다음 상태들의 가치 사이의 recursive relationship을 나타낸다.**
수식을 완전히 외우기보다는 이 의미를 알아야 해.
---
# 16. Backup Diagram
Backup은:
> 다음 상태의 value 정보를 현재 상태로 되돌려 전달하는 것
즉, 미래 상태의 가치를 이용해 현재 상태의 가치를 업데이트하는 느낌이야.
시험용 문장:
> **Backup은 successor state의 value 정보를 현재 state로 전달하는 과정이다.**
---
# 17. Gridworld 예제
Gridworld 예제는 Bellman equation이 어떻게 적용되는지 보여주는 예시야.
구성:
<table header-row="true">
<tr>
<td>요소</td>
<td>내용</td>
</tr>
<tr>
<td>Actions</td>
<td>north, south, east, west</td>
</tr>
<tr>
<td>Rewards</td>
<td>A에서 A′로 가면 +10, B에서 B′로 가면 +5, grid 밖으로 나가면 -1</td>
</tr>
<tr>
<td>Transition</td>
<td>일반 칸에서는 deterministic하게 한 칸 이동</td>
</tr>
<tr>
<td>Policy</td>
<td>equiprobable random policy</td>
</tr>
</table>
핵심:
> 각 action의 결과를 평균 내서 현재 state의 value를 계산한다.
즉, random policy라면 north/south/east/west를 각각 같은 확률로 선택한다고 보면 돼.
---
# 18. Optimal Policy와 Optimal Value Function
## Optimal policy
Optimal policy는:
> 모든 다른 policy보다 좋거나 같은 policy
자료 표현:
> policy π\\piπ가 π′\\pi'π′보다 좋다는 것은 모든 state에서 vπ(s)≥vπ′(s)라는 뜻
중요:
> optimal policy는 하나 이상 존재할 수 있다.
---
## Optimal value function
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>v∗(s)v\^\*(s)v∗(s)</td>
<td>모든 policy 중 가장 좋은 state value</td>
</tr>
<tr>
<td>q∗(s,a)q\^\*(s,a)q∗(s,a)</td>
<td>모든 policy 중 가장 좋은 action value</td>
</tr>
</table>
시험용 문장:
> **Optimal value function은 가능한 policy들 중 최대 value를 나타낸다.**
---
# 19. Bellman Optimality Equation
Bellman optimality equation은 Bellman equation에 **max**가 들어간 형태야.
일반 Bellman equation은 주어진 policy를 평가하는 식이고,
Bellman optimality equation은 최적 행동을 고르는 식이야.
핵심 차이:
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>Bellman equation</td>
<td>특정 policy π\\piπ의 value 계산</td>
</tr>
<tr>
<td>Bellman optimality equation</td>
<td>최적 policy의 value 계산</td>
</tr>
<tr>
<td>차이</td>
<td>max operation이 들어감</td>
</tr>
</table>
시험용 문장:
> **Bellman optimality equation은 각 state에서 가능한 action 중 최대 value를 선택한다.**
---
# 20. v∗v\^\*v∗와 q∗q\^\*q∗
자료에서 중요한 내용:
> v∗v\^\*v∗를 알면 optimal policy를 쉽게 구할 수 있다.
q∗q\^\*q∗를 알면 optimal action 선택이 더 쉬워진다.
왜냐하면 q∗(s,a)q\^\*(s,a)q∗(s,a)는 각 action의 가치를 직접 알려주기 때문이야.
시험용 문장:
> **q∗q\^\*q∗를 알면 각 state에서 가장 큰 q∗(s,a)q\^\*(s,a)q∗(s,a)를 갖는 action을 선택하면 된다.**
---
# 21. Optimality and Approximation
최적해를 정확히 구하려면 세 가지 가정이 필요해.
자료에 나온 3가지:
<table header-row="true">
<tr>
<td>가정</td>
<td>의미</td>
</tr>
<tr>
<td>1</td>
<td>환경의 dynamics를 정확히 알고 있어야 함</td>
</tr>
<tr>
<td>2</td>
<td>계산을 끝낼 충분한 computational resources가 있어야 함</td>
</tr>
<tr>
<td>3</td>
<td>Markov property가 성립해야 함</td>
</tr>
</table>
하지만 현실에서는 이 가정들이 어렵거나 불가능한 경우가 많아.
그래서:
> Bellman optimality equation을 근사하는 방법들이 필요하다.
그중 하나가 바로:
> Reinforcement learning
시험용 문장:
> **RL은 Bellman optimality equation을 근사적으로 해결하기 위한 방법 중 하나이다.**
---
# 22. Multi-period Optimization과 MDP
마지막 부분은 최적화 문제를 MDP로 바꾸는 관점이야.
예를 들어 재고 문제에서:
<table header-row="true">
<tr>
<td>MDP 요소</td>
<td>재고 문제에서의 의미</td>
</tr>
<tr>
<td>State</td>
<td>현재 재고 수준, 시간, 이전 주문 상태 등</td>
</tr>
<tr>
<td>Action</td>
<td>생산량 또는 주문량 결정</td>
</tr>
<tr>
<td>Reward</td>
<td>비용의 음수 또는 이익</td>
</tr>
<tr>
<td>Transition</td>
<td>현재 재고 + 생산/주문 - 수요 = 다음 재고</td>
</tr>
</table>
시험용 문장:
> **다기간 최적화 문제는 state, action, reward, transition을 정의하면 MDP로 표현할 수 있다.**
---
# 23. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 우선순위 낮아.
- Bellman equation 전체 수식 전개 과정
- Gridworld 숫자 계산 전체
- Recycling robot transition table 전체 암기
- 모든 확률식의 정확한 형태
- 마지막 inventory MIP 수식 전체
하지만 아래는 꼭 알아야 해.
- MDP 구성요소
- Agent-environment interface
- Markov property
- Return과 discounting
- Policy
- vπ(s)v_\\pi(s)vπ(s), qπ(s,a)
qπ(s,a)q_\\pi(s,a)
- Bellman equation 의미
- Bellman optimality equation 의미
- Optimal policy
- MDP와 강화학습의 관계
---
# 24. True/False 대비
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>MDP는 stochastic dynamic environment를 모델링한다.</td>
<td>True</td>
</tr>
<tr>
<td>Agent는 action을 선택하고 environment는 reward와 next state를 제공한다.</td>
<td>True</td>
</tr>
<tr>
<td>Markov property는 다음 상태가 전체 과거 history에만 의존한다는 뜻이다.</td>
<td>False</td>
</tr>
<tr>
<td>Markov property에서는 다음 상태와 보상이 현재 state와 action에만 의존한다.</td>
<td>True</td>
</tr>
<tr>
<td>Policy (\\pi(a</td>
<td>s))는 상태 sss에서 행동 aaa를 선택할 확률이다.</td>
</tr>
<tr>
<td>Return은 현재 이후의 reward sequence로부터 정의된다.</td>
<td>True</td>
</tr>
<tr>
<td>Discount rate γ\\gammaγ가 0에 가까우면 미래 reward를 더 많이 고려한다.</td>
<td>False</td>
</tr>
<tr>
<td>γ\\gammaγ가 1에 가까우면 더 먼 미래 reward도 중요하게 본다.</td>
<td>True</td>
</tr>
<tr>
<td>vπ(s)v_\\pi(s)vπ(s)는 policy π\\piπ 하에서 state sss의 가치이다.</td>
<td>True</td>
</tr>
<tr>
<td>qπ(s,a)q_\\pi(s,a)qπ(s,a)는 state-action pair의 가치이다.</td>
<td>True</td>
</tr>
<tr>
<td>Bellman equation은 현재 state value와 다음 state value 사이의 관계를 나타낸다.</td>
<td>True</td>
</tr>
<tr>
<td>Bellman optimality equation에는 max operation이 포함된다.</td>
<td>True</td>
</tr>
<tr>
<td>Optimal policy는 여러 개 존재할 수도 있다.</td>
<td>True</td>
</tr>
<tr>
<td>q∗q\^\*q∗를 알면 최적 action 선택이 쉬워진다.</td>
<td>True</td>
</tr>
<tr>
<td>RL은 MDP와 무관한 별개의 이론이다.</td>
<td>False</td>
</tr>
</table>

---

# 07] Temporal Difference Learning

---
## 목차
---
## 1. 이 PDF의 큰 주제
한마디로 말하면:
> **환경을 완전히 몰라도, 경험한 transition을 이용해서 Q-value를 학습하는 방법**
을 다루는 자료야.
여기서 경험한 transition은 이런 형태야.
(St,At,Rt+1,St+1)(S_t, A_t, R_\{t+1\}, S_\{t+1\})
(St,At,Rt+1,St+1)
또는 SARSA에서는 여기에 다음 행동까지 포함해서
(St,At,Rt+1,St+1,At+1)(S_t, A_t, R_\{t+1\}, S_\{t+1\}, A_\{t+1\})
(St,At,Rt+1,St+1,At+1)
를 사용해.
---
# 2. TD Learning이란?
## 핵심 의미
**TD Learning**은 **Temporal-Difference Learning**의 줄임말이야.
뜻은:
> 현재 추정값과 다음 상태를 보고 만든 새로운 추정값의 차이를 이용해 학습하는 방법
자료의 핵심 표현:
> TD can learn directly from raw experience.
TD updates estimates based on the Bellman equation.
즉, TD는 **MDP의 전체 transition probability를 몰라도**, 직접 경험한 샘플로 value를 업데이트해.
---
# 3. DP와 RL 차이
이 자료에서 중요한 복습 포인트야.
<table header-row="true">
<tr>
<td>구분</td>
<td>Dynamic Programming</td>
<td>Reinforcement Learning</td>
</tr>
<tr>
<td>환경 정보</td>
<td>p(s′,r∣s,a)p(s', r \\mid s,a)p(s′,r∣s,a)를 완전히 알고 있다고 가정</td>
<td></td>
</tr>
<tr>
<td>학습 방식</td>
<td>모든 상태/행동에 대해 계산</td>
<td></td>
</tr>
<tr>
<td>특징</td>
<td>정확하지만 환경 모델이 필요</td>
<td></td>
</tr>
<tr>
<td>RL</td>
<td>환경을 부분적으로만 알거나 몰라도 됨</td>
<td></td>
</tr>
<tr>
<td>RL 학습 방식</td>
<td>sample episode를 만들어 v∗v\^\*v∗, q∗q\^\*q∗를 추정</td>
<td></td>
</tr>
</table>
시험용 문장:
> **Dynamic programming은 환경의 dynamics를 알고 있다고 가정하지만, reinforcement learning은 sample experience를 통해 value를 추정한다.**
---
# 4. On-policy / Off-policy
이 자료에서 제일 중요한 개념 구분 중 하나야.
## 4-1. Behavior policy와 Target policy
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Behavior policy**</td>
<td>실제로 action을 선택해서 데이터를 만드는 policy</td>
</tr>
<tr>
<td>**Target policy**</td>
<td>우리가 학습하고 싶은 policy</td>
</tr>
</table>
---
## 4-2. On-policy
자료 표현:
> **behavior policy = target policy**
뜻:
> 행동하는 policy와 학습하는 policy가 같다.
대표 알고리즘:
> **SARSA**
시험용 문장:
> **SARSA는 on-policy TD control 방법이다.**
---
## 4-3. Off-policy
자료 표현:
> **behavior policy ≠ target policy**
뜻:
> 데이터를 만드는 policy와 학습하는 policy가 다르다.
대표 알고리즘:
> **Q-learning**
예를 들어 Q-learning에서는 실제 행동은 ϵ\\epsilonϵ-greedy로 탐험하면서,
학습 목표는 greedy policy 기준으로 잡을 수 있어.
시험용 문장:
> **Q-learning은 off-policy TD control 방법이다.**
---
# 5. ϵ\\epsilonϵ-greedy policy
## 의미
ϵ\\epsilonϵ-greedy는 exploration과 exploitation을 섞는 방법이야.
<table header-row="true">
<tr>
<td>개념</td>
<td>의미</td>
</tr>
<tr>
<td>**Exploitation**</td>
<td>지금 가장 좋아 보이는 action 선택</td>
</tr>
<tr>
<td>**Exploration**</td>
<td>다른 action도 시도해봄</td>
</tr>
</table>
ϵ\\epsilonϵ-greedy policy는:
> 대부분은 greedy action을 선택하고,
작은 확률 ϵ\\epsilonϵ로 다른 action도 선택하는 정책
자료에서는 모든 action이 최소한 어느 정도 선택될 수 있도록 하는 **soft policy**로 설명돼.
시험용 문장:
> **ϵ\\epsilonϵ-greedy policy는 대부분 greedy action을 선택하지만, ϵ\\epsilonϵ 확률로 exploration을 수행한다.**
---
# 6. SARSA
## 6-1. SARSA 이름의 의미
SARSA는 이 5개를 사용해서 업데이트하기 때문에 이름이 SARSA야.
St,At,Rt+1,St+1,At+1S_t, A_t, R_\{t+1\}, S_\{t+1\}, A_\{t+1\}
St,At,Rt+1,St+1,At+1
즉:
<table header-row="true">
<tr>
<td>글자</td>
<td>의미</td>
</tr>
<tr>
<td>S</td>
<td>현재 state</td>
</tr>
<tr>
<td>A</td>
<td>현재 action</td>
</tr>
<tr>
<td>R</td>
<td>reward</td>
</tr>
<tr>
<td>S</td>
<td>next state</td>
</tr>
<tr>
<td>A</td>
<td>next action</td>
</tr>
</table>
---
## 6-2. SARSA의 핵심
SARSA는:
> 현재 policy가 실제로 다음에 선택한 action At+1A_\{t+1\}At+1을 이용해서 Q-value를 업데이트한다.
업데이트 식은:
Q(St,At)←Q(St,At)+α[Rt+1+γQ(St+1,At+1)−Q(St,At)]Q(S_t,A_t) \\leftarrow Q(S_t,A_t) + \\alpha [R_\{t+1\} + \\gamma Q(S_\{t+1\},A_\{t+1\}) - Q(S_t,A_t)]
Q(St,At)←Q(St,At)+α[Rt+1+γQ(St+1,At+1)−Q(St,At)]
여기서 중요한 건 **다음 action At+1A_\{t+1\}At+1** 이 들어간다는 거야.
---
## 6-3. SARSA가 On-policy인 이유
SARSA는 실제로 행동을 고르는 policy와
업데이트에 사용하는 policy가 같아.
즉:
> 실제로 ϵ\\epsilonϵ-greedy로 행동을 선택하고,
업데이트도 그 ϵ\\epsilonϵ-greedy가 선택한 다음 행동을 기준으로 한다.
그래서 on-policy야.
시험용 문장:
> **SARSA는 실제로 선택한 다음 action At+1A_\{t+1\}At+1을 사용하여 업데이트하므로 on-policy이다.**
---
# 7. Q-learning
## 7-1. Q-learning의 핵심
Q-learning은:
> 다음 state에서 실제로 어떤 action을 선택했는지와 상관없이, 가장 큰 Q값을 기준으로 업데이트한다.
업데이트 식은:
Q(St,At)←Q(St,At)+α[Rt+1+γmax⁡aQ(St+1,a)−Q(St,At)]Q(S_t,A_t) \\leftarrow Q(S_t,A_t) + \\alpha [R_\{t+1\} + \\gamma \\max_a Q(S_\{t+1\},a) - Q(S_t,A_t)]
Q(St,At)←Q(St,At)+α[Rt+1+γamaxQ(St+1,a)−Q(St,At)]
여기서 중요한 건:
max⁡aQ(St+1,a)\\max_a Q(S_\{t+1\},a)
amaxQ(St+1,a)
이 부분이야.
---
## 7-2. Q-learning이 Off-policy인 이유
Q-learning은 실제 행동은 ϵ\\epsilonϵ-greedy로 탐험하면서 선택할 수 있어.
하지만 업데이트할 때는:
> 다음 state에서 가장 좋은 action을 선택한다고 가정
해.
즉, 실제 행동 policy와 학습 목표 policy가 달라.
그래서 off-policy야.
시험용 문장:
> **Q-learning은 behavior policy와 target policy가 다르기 때문에 off-policy이다.**
---
# 8. SARSA vs Q-learning
이 비교가 시험에 나오기 좋아.
<table header-row="true">
<tr>
<td>구분</td>
<td>SARSA</td>
<td>Q-learning</td>
</tr>
<tr>
<td>분류</td>
<td>On-policy</td>
<td>Off-policy</td>
</tr>
<tr>
<td>업데이트에 쓰는 다음 값</td>
<td>실제 선택한 At+1A_\{t+1\}At+1</td>
<td>가능한 action 중 최대값</td>
</tr>
<tr>
<td>필요한 샘플</td>
<td>S,A,R,S′,A′S,A,R,S',A'S,A,R,S′,A′</td>
<td>S,A,R,S′S,A,R,S'S,A,R,S′</td>
</tr>
<tr>
<td>학습 대상</td>
<td>현재 policy의 qπq_\\piqπ</td>
<td>최적 q∗q\^\*q∗</td>
</tr>
<tr>
<td>특징</td>
<td>실제 탐험 policy를 반영</td>
<td>greedy target을 직접 학습</td>
</tr>
</table>
핵심 문장:
> **SARSA는 실제로 다음에 선택한 action을 사용하고, Q-learning은 다음 state에서 최대 Q값을 사용한다.**
---
# 9. Learning rate α\\alphaα, Discount rate γ\\gammaγ
## α\\alphaα, step size
α\\alphaα는 학습률이야.
<table header-row="true">
<tr>
<td>값</td>
<td>의미</td>
</tr>
<tr>
<td>α\\alphaα 큼</td>
<td>새 정보 반영을 크게 함</td>
</tr>
<tr>
<td>α\\alphaα 작음</td>
<td>천천히 업데이트함</td>
</tr>
</table>
시험용 문장:
> **α\\alphaα는 Q-value를 얼마나 크게 업데이트할지 정하는 step size이다.**
---
## γ\\gammaγ, discount rate
γ\\gammaγ는 미래 reward를 얼마나 중요하게 볼지 정하는 값이야.
<table header-row="true">
<tr>
<td>값</td>
<td>의미</td>
</tr>
<tr>
<td>γ→0\\gamma \\to 0γ→0</td>
<td>당장 reward 중심</td>
</tr>
<tr>
<td>γ→1\\gamma \\to 1γ→1</td>
<td>미래 reward도 중요하게 고려</td>
</tr>
</table>
---
# 10. TD Error
SARSA와 Q-learning 식에서 대괄호 안쪽이 중요해.
[image omitted: temporary Notion asset]
SARSA 기준:
Rt+1+γQ(St+1,At+1)−Q(St,At)R_\{t+1\} + \\gamma Q(S_\{t+1\}, A_\{t+1\}) - Q(S_t,A_t)
Rt+1+γQ(St+1,At+1)−Q(St,At)
Q-learning 기준:
Rt+1+γmax⁡aQ(St+1,a)−Q(St,At)R_\{t+1\} + \\gamma \\max_a Q(S_\{t+1\},a) - Q(S_t,A_t)
Rt+1+γamaxQ(St+1,a)−Q(St,At)
이 차이가 바로 **TD error**라고 보면 돼.
쉽게 말하면:
> 새로 계산한 목표값과 현재 Q 추정값의 차이
TD error가 크면 현재 추정이 많이 틀린 거고,
그만큼 Q값을 업데이트해.
시험용 문장:
> **TD error는 target estimate와 old estimate의 차이이다.**
---
# 11. Windy Gridworld 예제
자료의 예제야.
## 문제 구조
<table header-row="true">
<tr>
<td>요소</td>
<td>내용</td>
</tr>
<tr>
<td>환경</td>
<td>바람이 위쪽으로 부는 gridworld</td>
</tr>
<tr>
<td>Action</td>
<td>Up, Down, Right, Left</td>
</tr>
<tr>
<td>Reward</td>
<td>goal에 도달할 때까지 -1</td>
</tr>
<tr>
<td>알고리즘</td>
<td>ϵ\\epsilonϵ-greedy SARSA</td>
</tr>
<tr>
<td>목적</td>
<td>goal까지 더 빨리 도달하는 policy 학습</td>
</tr>
</table>
핵심:
> 시간이 지날수록 goal에 더 빨리 도달하면 학습이 잘 되고 있다는 뜻이다.
자료에서는 slope가 증가하는 것이 goal에 더 빨리 도달함을 보여준다고 설명해.
---
# 12. Maximization Bias
## 의미
Q-learning은 업데이트할 때 max를 사용하지?
max⁡aQ(St+1,a)\\max_a Q(S_\{t+1\},a)
amaxQ(St+1,a)
문제는 Q값들이 아직 정확하지 않은 **추정값**이라는 거야.
그런데 그 추정값들 중 가장 큰 값을 고르면,
우연히 과대평가된 action이 선택될 수 있어.
이게 **maximization bias**야.
시험용 문장:
> **Maximization bias는 추정된 값들 중 최대값을 사용하면서 실제보다 value를 과대평가할 수 있는 현상이다.**
---
# 13. Double Learning
## 왜 필요한가?
Maximization bias를 줄이기 위해 나온 아이디어야.
문제는:
> 같은 Q값으로 action도 고르고, 그 action의 value도 평가한다는 것
이야.
Double learning은 이 둘을 분리해.
---
## 핵심 아이디어
두 개의 Q 추정값을 둬.
Q1,Q2Q_1, Q_2
Q1,Q2
하나는 action 선택에 사용하고,
다른 하나는 value 평가에 사용해.
예를 들어:
<table header-row="true">
<tr>
<td>역할</td>
<td>사용</td>
</tr>
<tr>
<td>Q1Q_1Q1</td>
<td>어떤 action이 좋아 보이는지 선택</td>
</tr>
<tr>
<td>Q2Q_2Q2</td>
<td>그 action의 value를 평가</td>
</tr>
</table>
시험용 문장:
> **Double learning은 action 선택과 value 평가를 분리하여 maximization bias를 줄이려는 방법이다.**
---
# 14. Double Q-learning
Double Q-learning은:
> Double learning 아이디어를 Q-learning에 적용한 방법
핵심:
- Q1Q_1Q1, Q2 두 개를 학습한다.
Q2Q_2
- 매번 둘 중 하나만 업데이트한다.
- action 선택과 평가를 서로 다른 Q 함수로 나눈다.
- behavior policy는 Q1과 Q2의 평균 등을 이용할 수 있다.
Q1Q_1
Q2Q_2
시험용 문장:
> **Double Q-learning은 두 개의 Q-value estimate를 사용하여 maximization bias를 줄인다.**
---
# 15. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 낮은 우선순위야.
- SARSA pseudo-code 전체 문장
- Q-learning pseudo-code 전체 문장
- Double Q-learning 업데이트 식 전체 암기
- Windy Gridworld 그래프 숫자
- Maximization bias 예제 그림 세부 구조
하지만 아래는 꼭 알아야 해.
- TD learning의 의미
- On-policy / Off-policy 차이
- SARSA와 Q-learning 차이
- ϵ\\epsilonϵ-greedy policy
- α\\alphaα, γ
γ\\gamma
- Maximization bias
- Double learning / Double Q-learning
---
# 16. True/False 대비
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>TD learning은 raw experience로부터 직접 학습할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>TD learning은 Bellman equation을 기반으로 estimate를 업데이트한다.</td>
<td>True</td>
</tr>
<tr>
<td>On-policy에서는 behavior policy와 target policy가 같다.</td>
<td>True</td>
</tr>
<tr>
<td>Off-policy에서는 behavior policy와 target policy가 다르다.</td>
<td>True</td>
</tr>
<tr>
<td>SARSA는 off-policy 방법이다.</td>
<td>False</td>
</tr>
<tr>
<td>SARSA는 S,A,R,S′,A′S,A,R,S',A'S,A,R,S′,A′를 사용한다.</td>
<td>True</td>
</tr>
<tr>
<td>Q-learning은 on-policy 방법이다.</td>
<td>False</td>
</tr>
<tr>
<td>Q-learning은 다음 state에서 최대 Q값을 사용한다.</td>
<td>True</td>
</tr>
<tr>
<td>Q-learning은 S,A,R,S′S,A,R,S'S,A,R,S′만으로 업데이트할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>ϵ\\epsilonϵ-greedy는 exploration과 exploitation을 함께 고려한다.</td>
<td>True</td>
</tr>
<tr>
<td>α\\alphaα는 discount rate이다.</td>
<td>False</td>
</tr>
<tr>
<td>γ\\gammaγ는 미래 reward의 중요도를 조절한다.</td>
<td>True</td>
</tr>
<tr>
<td>Maximization bias는 max 연산 때문에 value를 과대평가할 수 있는 현상이다.</td>
<td>True</td>
</tr>
<tr>
<td>Double Q-learning은 maximization bias를 줄이기 위해 두 개의 Q estimate를 사용한다.</td>
<td>True</td>
</tr>
</table>
---
# 최종 암기 요약
이 자료는 아래 문장들만 확실히 잡으면 돼.
> **TD learning은 Bellman equation을 기반으로, 실제 경험 transition을 이용해 value를 업데이트하는 방법이다.**
> **On-policy는 행동하는 policy와 학습하는 policy가 같고, off-policy는 둘이 다르다.**
> **SARSA는 on-policy TD control이고, S,A,R,S′,A′S,A,R,S',A'S,A,R,S′,A′를 사용한다.**
> **Q-learning은 off-policy TD control이고, 다음 state에서 max⁡aQ(S′,a)\\max_a Q(S',a)maxaQ(S′,a)를 사용한다.**
> **ϵ\\epsilonϵ-greedy는 대부분 greedy action을 선택하지만, 일부 확률로 exploration을 수행한다.**
> **SARSA는 실제로 선택한 다음 action을 반영하고, Q-learning은 다음 state에서 가장 좋아 보이는 action을 기준으로 업데이트한다.**
> **Maximization bias는 max 연산으로 인해 Q값이 과대평가될 수 있는 현상이다.**
> **Double Q-learning은 두 개의 Q estimate를 사용해 action 선택과 value 평가를 분리함으로써 maximization bias를 줄인다.**
# 1. MDP를 몰라도 TD Learning으로 풀 수 있다는 뜻이야?
정확히는 **아니야.**
TD Learning도 기본적으로는 MDP 구조가 필요해.
즉, 최소한 이런 건 정해야 해.
<table header-row="true">
<tr>
<td>MDP 요소</td>
<td>예시: 재고 문제</td>
</tr>
<tr>
<td>State</td>
<td>현재 재고</td>
</tr>
<tr>
<td>Action</td>
<td>주문량</td>
</tr>
<tr>
<td>Reward</td>
<td>비용의 음수</td>
</tr>
<tr>
<td>Next state</td>
<td>다음 재고</td>
</tr>
</table>
그런데 TD Learning은 **MDP의 transition probability 전체를 몰라도 된다**는 뜻이야.
즉, 이런 확률표를 몰라도 돼.
p(s′,r∣s,a)p(s', r \\mid s, a)
p(s′,r∣s,a)
이건 뭐냐면:
> 현재 상태 sss에서 행동 aaa를 했을 때,
다음 상태가 s′s's′가 되고 reward가 rrr이 나올 확률
예를 들어 재고가 10개이고 주문을 5개 했을 때,
<table header-row="true">
<tr>
<td>다음 재고</td>
<td>보상</td>
<td>확률</td>
</tr>
<tr>
<td>15</td>
<td>-10</td>
<td>0.2</td>
</tr>
<tr>
<td>12</td>
<td>-20</td>
<td>0.5</td>
</tr>
<tr>
<td>8</td>
<td>-50</td>
<td>0.3</td>
</tr>
</table>
이런 식의 **전체 확률표**를 알아야 하는 게 DP야.
그런데 RL/TD Learning은 이렇게 안 해.
그냥 실제로 한 번 해보고:
> 재고 10 → 주문 5 → 수요 발생 → 다음 재고 12 → 비용 20
이런 **sample transition** 하나를 보고 업데이트해.
그래서 핵심은:
> **TD Learning은 MDP를 안 쓰는 게 아니라, MDP의 확률모델을 정확히 몰라도 sample 경험으로 학습하는 방법이다.**
---
# 2. 갑자기 DP랑 RL은 왜 나온 거야?
교수님 자료에서는 “Bellman optimality equation을 어떻게 풀 것인가?”라는 흐름에서 DP와 RL을 비교한 거야.
자료에서도 optimal policy를 찾기 위해 Bellman optimality equation을 풀어야 하는데, 직접 풀기 어렵고, 접근법으로 **Dynamic Programming과 Reinforcement Learning**이 있다고 설명해.
## Dynamic Programming, DP
DP는:
> 환경의 dynamics, 즉 p(s′,r∣s,a)p(s',r\|s,a)p(s′,r∣s,a)를 완전히 알고 있을 때 계산으로 푸는 방법
이야.
즉, 모든 상태와 행동에 대해:
> 이 행동을 하면 어떤 다음 상태로 갈 확률이 얼마인지
어떤 reward가 나올 확률이 얼마인지
를 알고 있다고 가정해.
그래서 DP는 **모델을 알고 푸는 planning 방법**에 가까워.
---
## Reinforcement Learning, RL
RL은:
> 환경의 transition probability를 정확히 몰라도, 직접 경험하면서 학습하는 방법
이야.
즉, 확률표 전체를 몰라도 agent가 행동해보고, reward와 next state를 관찰하면서 v∗v\^\*v∗나 q∗q\^\*q∗를 추정해.
---
# 3. 우리가 Dynamic Programming을 다룬 적 있었나?
깊게 다루지는 않았어.
이 자료에서는 DP를 자세히 배우는 게 아니라, **RL과 비교하기 위해 언급한 것**에 가까워.
즉, 교수님이 말하고 싶은 건 이거야.
> Bellman equation을 정확히 풀려면 DP 같은 방법이 있지만,
DP는 환경 모델을 알아야 한다.
그런데 현실에서는 모델을 모르는 경우가 많으니까 RL을 쓴다.
그래서 네가 시험 대비로 알아야 할 정도는 이거야.
<table header-row="true">
<tr>
<td>구분</td>
<td>Dynamic Programming, DP</td>
<td>Reinforcement Learning, RL</td>
</tr>
<tr>
<td>환경 모델</td>
<td>(p(s', r \\mid s,a))를 완전히 알고 있다고 가정</td>
<td>(p(s', r \\mid s,a))를 몰라도 됨</td>
</tr>
<tr>
<td>학습/계산 방식</td>
<td>모든 state/action에 대해 계산으로 value를 구함</td>
<td>실제 경험 sample을 이용해 value를 추정함</td>
</tr>
<tr>
<td>필요한 것</td>
<td>MDP의 transition probability, reward 구조</td>
<td>경험 데이터: (S, A, R, S')</td>
</tr>
<tr>
<td>대표 방식</td>
<td>Value iteration, Policy iteration</td>
<td>SARSA, Q-learning, DQN 등</td>
</tr>
<tr>
<td>특징</td>
<td>정확하지만 환경 모델이 필요함</td>
<td>모델을 몰라도 학습 가능하지만 sample이 많이 필요함</td>
</tr>
</table>
DP는 강화학습 알고리즘이라기보다는, **MDP를 푸는 고전적인 계산 방법**이야.
RL은 **경험으로 MDP를 푸는 학습 방법**이고.
---
# 4. 그럼 이제부터가 강화학습이야?
거의 맞아.
흐름은 이렇게 보면 돼.
> **MDP**: 강화학습 문제를 수학적으로 정의하는 틀
**Bellman equation**: MDP에서 value를 계산하는 기본 관계식
**TD Learning**: Bellman equation을 경험 데이터로 근사해서 학습하는 강화학습 방법
즉, MDP도 강화학습에서 계속 사용해.
강화학습에서도 문제를 풀려면 반드시 이런 걸 정해야 해.
<table header-row="true">
<tr>
<td>요소</td>
<td>예시</td>
</tr>
<tr>
<td>State</td>
<td>현재 상황</td>
</tr>
<tr>
<td>Action</td>
<td>선택 가능한 행동</td>
</tr>
<tr>
<td>Reward</td>
<td>행동 결과</td>
</tr>
<tr>
<td>Transition</td>
<td>상태가 바뀌는 방식</td>
</tr>
</table>
다만 RL은 transition probability를 정확히 몰라도 된다는 점이 달라.
---
# 5. Behavior policy랑 Target policy가 뭐야?
이건 엄청 중요해.
## Behavior policy
**Behavior policy**는 실제로 agent가 행동할 때 사용하는 policy야.
즉,
> 데이터를 만들기 위해 실제로 action을 선택하는 정책
이야.
예를 들어 agent가 학습 중에 ϵ\\epsilonϵ-greedy로 행동한다면,
그 ϵ\\epsilonϵ-greedy policy가 behavior policy야.
---
## Target policy
**Target policy**는 우리가 최종적으로 배우고 싶은 policy야.
즉,
> 업데이트가 목표로 삼는 정책
이야.
예를 들어 Q-learning에서는 실제 행동은 ϵ\\epsilonϵ-greedy로 하더라도,
업데이트할 때는 “다음 상태에서 가장 Q값이 큰 행동”을 기준으로 해.
그러면 target policy는 greedy policy야.
---
# 6. 실제 행동 policy와 업데이트 policy가 왜 나뉘어?
탐험해야 하기 때문이야.
학습 중에는 agent가 항상 제일 좋아 보이는 action만 하면 안 돼.
가끔은 다른 action도 해봐야 해.
그래서 실제 행동은:
> ϵ\\epsilonϵ-greedy로 탐험 포함
하지만 학습 목표는:
> 결국 greedy하게 가장 좋은 action을 선택하는 policy
일 수 있어.
이렇게 실제 행동하는 policy와 학습 목표 policy가 달라지면 **off-policy**야.
---
# 7. SARSA와 Q-learning에서 behavior / target은 뭐야?
## SARSA
SARSA는 **on-policy**야.
<table header-row="true">
<tr>
<td>구분</td>
<td>SARSA</td>
</tr>
<tr>
<td>Behavior policy</td>
<td>ϵ\\epsilonϵ-greedy</td>
</tr>
<tr>
<td>Target policy</td>
<td>같은 ϵ\\epsilonϵ-greedy</td>
</tr>
<tr>
<td>특징</td>
<td>실제로 다음에 선택한 action At+1A_\{t+1\}At+1을 업데이트에 사용</td>
</tr>
</table>
즉, SARSA는 실제로 행동한 그대로 배워.
그래서:
> 행동하는 policy = 학습하는 policy
---
## Q-learning
Q-learning은 **off-policy**야.
<table header-row="true">
<tr>
<td>구분</td>
<td>Q-learning</td>
</tr>
<tr>
<td>Behavior policy</td>
<td>보통 ϵ\\epsilonϵ-greedy</td>
</tr>
<tr>
<td>Target policy</td>
<td>greedy policy</td>
</tr>
<tr>
<td>특징</td>
<td>실제 다음 행동과 상관없이 max⁡aQ(St+1,a)\\max_a Q(S_\{t+1\},a)maxaQ(St+1,a) 사용</td>
</tr>
</table>
즉, Q-learning은 실제로는 탐험하면서 행동하지만,
업데이트는 “최적 행동을 한다면?” 기준으로 해.
그래서:
> 행동하는 policy ≠ 학습하는 policy
---
# 8. old estimate는 뭐야?
old estimate는 **업데이트하기 전의 기존 Q값**이야.
예를 들어 현재 우리가 이렇게 알고 있다고 해보자.
> Q(s,a)=10Q(s,a) = 10Q(s,a)=10
그런데 실제로 경험해보니 reward도 받고 다음 상태도 보니까
새로운 목표값이 14처럼 보였어.
그러면 Q값을 바로 14로 바꾸는 게 아니라,
조금만 이동시켜.
예:
> old estimate: 10
new target: 14
updated estimate: 11 또는 12 정도
즉, TD Learning은 기존 추정값을 새로운 경험을 반영해서 조금 수정하는 방식이야.
---
# 9. ϵ\\epsilonϵ-greedy SARSA는 뭐야?
그냥 **SARSA인데 action 선택을 ϵ\\epsilonϵ-greedy policy로 하는 것**이야.
SARSA 자체는 업데이트 방식이고,
ϵ\\epsilonϵ-greedy는 action을 고르는 방식이야.
즉:
> SARSA = Q값을 업데이트하는 알고리즘
ϵ\\epsilonϵ-greedy = 행동을 선택하는 정책
그래서 ϵ\\epsilonϵ-greedy SARSA는:
> action은 ϵ\\epsilonϵ-greedy로 고르고,
업데이트는 SARSA 방식으로 하는 알고리즘
이야.
---
# 10. Q-learning에서 q는 그 action-value function 맞아?
응, 맞아.
Q-learning의 Q는 q(s,a)q(s,a)q(s,a), 즉 **action-value function**을 추정하는 거야.
질문은 이거야.
> 상태 sss에서 행동 aaa를 하면 앞으로 얼마나 좋을까?
그래서 Q-learning은 결국:
> q∗(s,a)q\^\*(s,a)q∗(s,a), 즉 최적 action-value function을 학습하는 방법
이라고 보면 돼.
---
# 11. max Q를 찾는 게 목표인데, 왜 과대평가가 문제야?
좋은 질문이야.
Q-learning은 다음 상태에서 가장 큰 Q값을 사용해.
max⁡aQ(St+1,a)\\max_a Q(S_\{t+1\},a)
amaxQ(St+1,a)
문제는 학습 초반의 Q값은 **진짜 값이 아니라 추정값**이라는 거야.
예를 들어 실제로는 세 action의 진짜 가치가 모두 0이라고 해보자.
그런데 sample이 우연히 이렇게 추정될 수 있어.
<table header-row="true">
<tr>
<td>action</td>
<td>실제 가치</td>
<td>현재 추정 Q</td>
</tr>
<tr>
<td>A</td>
<td>0</td>
<td>-1</td>
</tr>
<tr>
<td>B</td>
<td>0</td>
<td>0.5</td>
</tr>
<tr>
<td>C</td>
<td>0</td>
<td>2</td>
</tr>
</table>
그러면 Q-learning은 C를 고르겠지.
왜냐하면 C의 Q가 제일 크니까.
그런데 C가 진짜 좋은 게 아니라,
**우연히 추정값이 높게 나온 것**일 수 있어.
이게 maximization bias야.
> max는 여러 추정값 중에서 가장 큰 값을 고르기 때문에, 우연히 과대평가된 값을 선택하기 쉽다.
---
# 12. “같은 Q로 action도 고르고 value도 평가한다”는 게 무슨 뜻이야?
네가 이해한 게 맞아.
Q-learning에서는 같은 Q값을 이용해서 두 가지를 동시에 해.
1. **어떤 action이 좋아 보이는지 선택**
2. **그 action의 value를 평가**
예를 들어 다음 상태에서:
<table header-row="true">
<tr>
<td>action</td>
<td>Q estimate</td>
</tr>
<tr>
<td>A</td>
<td>1</td>
</tr>
<tr>
<td>B</td>
<td>5</td>
</tr>
<tr>
<td>C</td>
<td>3</td>
</tr>
</table>
이면 Q-learning은 B를 선택해.
그리고 동시에:
> B의 value는 5다
라고 평가해.
즉, 같은 Q가:
> “B가 제일 좋아!”
“그리고 B의 가치는 5야!”
를 둘 다 말하는 거야.
문제는 이 Q가 아직 부정확하면,
우연히 높게 나온 값을 스스로 선택하고 스스로 믿어버릴 수 있어.
그래서 Double learning이 나온 거야.
---
# 13. Double learning은 정확히 뭘 분리해?
Double learning은 두 개의 Q를 둬.
Q1, Q2Q_1,\\ Q_2
Q1, Q2
그리고 역할을 나눠.
예를 들어:
<table header-row="true">
<tr>
<td>역할</td>
<td>사용</td>
</tr>
<tr>
<td>action 선택</td>
<td>Q1Q_1Q1</td>
</tr>
<tr>
<td>선택된 action의 value 평가</td>
<td>Q2Q_2Q2</td>
</tr>
</table>
즉,
> Q1Q_1Q1이 “B가 좋아 보인다”고 선택하면,
Q2Q_2Q2가 “B의 가치는 이 정도다”라고 평가해.
이렇게 하면 한 Q가 자기 혼자 고르고 자기 혼자 평가하는 문제를 줄일 수 있어.
그래서 네 말처럼:
> Q가 자기 자신이 선택한 action을 자기 자신이 평가해서 생기는 과대평가를 줄이기 위해 Double learning이 나온 것
맞아.
---
# 14. 왜 TD Learning이 Bellman equation 기반이야?
Bellman equation은 MDP에서 value가 만족해야 하는 관계야.
예를 들어 Q-value는 기본적으로 이런 관계를 가져.
> 현재 Q값
= 지금 받은 reward
- 다음 상태의 미래 가치
TD Learning은 이걸 그대로 사용해.
다만 Bellman equation에서는 원래 모든 가능한 다음 상태를 확률로 평균내야 해.
즉, 원래는:
> 모든 s′s's′, 모든 reward 가능성을 다 고려해서 평균 계산
해야 해.
그런데 TD Learning은 그걸 하지 않고,
실제로 경험한 sample 하나로 근사해.
예를 들어 원래 Bellman은:
> 이 action을 하면 가능한 모든 다음 상태를 평균내자
TD Learning은:
> 이번에 실제로 가본 다음 상태 하나를 보고 업데이트하자
야.
그래서 TD Learning이 Bellman equation 기반이라는 말은:
> Bellman equation의 “현재 가치 = 즉시 reward + 다음 가치” 구조를 사용하되,
전체 확률 평균 대신 실제 sample transition으로 업데이트한다
는 뜻이야.
---
# 15. Bellman은 MDP 푸는 방정식 아니야?
맞아.
Bellman equation은 MDP에서 value function을 구하는 핵심 방정식이야.
그런데 이걸 푸는 방법이 여러 개야.
<table header-row="true">
<tr>
<td>방법</td>
<td>설명</td>
</tr>
<tr>
<td>DP</td>
<td>transition probability를 알고 Bellman equation을 계산으로 품</td>
</tr>
<tr>
<td>TD Learning</td>
<td>transition probability를 몰라도 sample experience로 Bellman 구조를 근사</td>
</tr>
<tr>
<td>Q-learning</td>
<td>TD 방식으로 q∗q\^\*q∗를 학습</td>
</tr>
<tr>
<td>SARSA</td>
<td>TD 방식으로 현재 policy의 qπq_\\piqπ를 학습</td>
</tr>
</table>
즉,
> Bellman equation은 기본 원리
TD Learning은 그 원리를 sample로 적용하는 학습 방법
이야.
---
# 16. 전체 흐름을 한 번에 보면
이렇게 이해하면 돼.
> MDP는 state, action, reward, transition으로 문제를 정의한다.<br><br>Bellman equation은 MDP에서 value가 만족해야 하는 관계식이다.<br><br>DP는 transition probability를 다 알고 Bellman equation을 계산으로 푼다.<br><br>RL은 transition probability를 몰라도 경험 sample로 value를 학습한다.<br><br>TD Learning은 Bellman equation의 구조를 이용해서 sample transition마다 Q값을 업데이트한다.<br><br>SARSA는 실제 선택한 다음 action을 반영하므로 on-policy다.<br><br>Q-learning은 다음 state에서 max Q를 사용하므로 off-policy다.<br><br>max 연산은 과대평가를 만들 수 있고, Double Q-learning은 이를 줄이기 위해 두 Q를 사용한다.

---

# 08] Deep Q-Network

---
## 목차
---
## 1. 이 PDF의 큰 주제
이 자료는 한마디로:
> **Q-learning + Deep Learning = DQN**
을 설명하는 자료야.
앞에서 Q-learning은 Q(s,a)Q(s,a)Q(s,a) 값을 table에 저장했어.
그런데 state가 너무 많거나 복잡하면 table로 저장할 수 없어.
예를 들어:
- 이미지 입력
- 재고 상태가 매우 다양함
- action 조합이 많음
- 상태가 연속형에 가까움
이런 경우에는 모든 Q(s,a)Q(s,a)Q(s,a)를 표로 저장하기 어렵지.
그래서 DQN에서는:
> Q(s,a)Q(s,a)Q(s,a)를 table에 저장하지 않고, neural network가 예측하게 만든다.
---
# 2. Machine Learning / Deep Learning / RL 구분
## Machine Learning
Machine Learning은 데이터로부터 패턴을 학습하는 방법이야.
대표적으로:
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>Supervised Learning</td>
<td>정답 label이 있는 데이터로 학습</td>
</tr>
<tr>
<td>Unsupervised Learning</td>
<td>정답 없이 데이터 구조나 패턴을 찾음</td>
</tr>
<tr>
<td>Reinforcement Learning</td>
<td>환경과 상호작용하면서 좋은 policy를 학습</td>
</tr>
</table>
---
## Deep Learning
Deep Learning은 neural network를 사용해서 복잡한 feature를 자동으로 추출하는 방법이야.
자료에서 중요한 차이는 이거야.
<table header-row="true">
<tr>
<td>구분</td>
<td>특징</td>
</tr>
<tr>
<td>일반 Machine Learning</td>
<td>사람이 직접 feature를 설계해야 함</td>
</tr>
<tr>
<td>Deep Learning</td>
<td>neural network가 feature를 자동으로 학습</td>
</tr>
</table>
예를 들어 이미지에서 물체를 분류할 때,
사람이 직접 “모서리, 색상, 질감” 같은 feature를 정하지 않아도
deep neural network가 알아서 feature를 뽑아내는 거야.
---
# 3. Deep Reinforcement Learning, DRL
DRL은:
> **Deep Learning + Reinforcement Learning**
이야.
즉,
> 강화학습 문제에서 value function이나 policy를 neural network로 근사하는 방법
이라고 보면 돼.
DRL이 필요한 이유는:
> state space나 action space가 너무 커서 table 방식 RL을 쓰기 어렵기 때문
자료에서는 게임, 로봇 제어, 금융, 자율주행 같은 예시가 나와.
---
# 4. RL vs DRL
## RL, Tabular method
기존 RL에서는 Q(s,a)Q(s,a)Q(s,a)를 table로 저장해.
<table header-row="true">
<tr>
<td>State</td>
<td>Action 1</td>
<td>Action 2</td>
<td>Action 3</td>
</tr>
<tr>
<td>s1s_1s1</td>
<td>3.2</td>
<td>1.5</td>
<td>0.7</td>
</tr>
<tr>
<td>s2s_2s2</td>
<td>2.1</td>
<td>4.8</td>
<td>1.0</td>
</tr>
</table>
이런 방식이야.
대표 알고리즘:
- SARSA
- Q-learning
하지만 state/action이 커지면 table이 너무 커져서 불가능해.
---
## DRL, Function approximation
DRL에서는 table 대신 neural network를 사용해.
즉:
Q(s,a)≈Qθ(s,a)Q(s,a) \\approx Q_\\theta(s,a)
Q(s,a)≈Qθ(s,a)
여기서 θ\\thetaθ는 neural network의 parameter야.
쉽게 말하면:
> 정확한 Q-table을 저장하는 대신, 신경망이 Q값을 예측하게 한다.
---
# 5. Value approximation / Policy approximation
자료에서는 approximation을 두 종류로 나눠.
## 5-1. Value space approximation
가치함수를 근사하는 방식이야.
예:
Q(s,a)≈Qθ(s,a)Q(s,a) \\approx Q_\\theta(s,a)
Q(s,a)≈Qθ(s,a)
대표 알고리즘:
- DQN
- Dueling DQN
핵심:
> state와 action을 넣으면 Q-value를 예측한다.
---
## 5-2. Policy space approximation
policy 자체를 근사하는 방식이야.
예:
π(a∣s)≈πθ(a∣s)\\pi(a\|s) \\approx \\pi_\\theta(a\|s)
π(a∣s)≈πθ(a∣s)
대표 알고리즘:
- REINFORCE
- TRPO
- PPO
핵심:
> state를 넣으면 action을 선택할 확률을 직접 출력한다.
---
# 6. DQN이란?
자료의 핵심 문장:
> **DQN = Q-learning + Deep Learning**
즉,
<table header-row="true">
<tr>
<td>구성</td>
<td>역할</td>
</tr>
<tr>
<td>Q-learning</td>
<td>최적 policy를 찾는 강화학습 방법</td>
</tr>
<tr>
<td>Deep Neural Network</td>
<td>복잡한 state/action space에서 Q-value를 근사</td>
</tr>
</table>
DQN은 결국 Q-learning인데,
Q(s,a)Q(s,a)Q(s,a)를 table 대신 neural network Qθ(s,a)Q_\\theta(s,a)Qθ(s,a)로 표현하는 거야.
---
# 7. DQN은 어떤 문제에 쓰나?
자료에서는 DQN 구조를 이렇게 설명해.
> **Large continuous state space + discrete action space**
즉:
<table header-row="true">
<tr>
<td>요소</td>
<td>DQN에서 적합한 경우</td>
</tr>
<tr>
<td>State</td>
<td>크거나 연속적인 상태 가능</td>
</tr>
<tr>
<td>Action</td>
<td>discrete action이어야 함</td>
</tr>
</table>
예를 들어 Atari 게임에서는 화면 이미지가 state이고,
action은 위/아래/왼쪽/오른쪽/발사 같은 discrete action이야.
중요:
> DQN은 action이 연속형이면 적용하기 어렵다.
그래서 continuous action 문제에서는 policy-based algorithm이 필요해질 수 있어.
---
# 8. Naive DQN
Naive DQN은 단순히 Q-learning의 Q-table을 neural network로 바꾼 형태야.
흐름은 이거야.
1. ϵ\\epsilonϵ-greedy로 action 선택
2. transition (s,a,r,s′) 관찰
(s,a,r,s′)(s,a,r,s')
3. target 계산
4. loss 계산
5. gradient descent로 network 업데이트
그런데 자료에서는 답이 **No**라고 해.
즉:
> 단순히 Q-table을 neural network로 바꾸는 것만으로는 충분하지 않다.
---
# 9. Naive DQN의 문제점
자료에서 두 가지 문제가 나와.
## 9-1. Samples are correlated
강화학습 데이터는 시간 순서대로 생성돼.
예를 들어:
s1→s2→s3→s4s_1 \\rightarrow s_2 \\rightarrow s_3 \\rightarrow s_4
s1→s2→s3→s4
이렇게 연속된 데이터는 서로 강하게 연결되어 있어.
즉, sample들이 독립적이지 않아.
이게 학습을 불안정하게 만들 수 있어.
---
## 9-2. Target value is changing
DQN에서는 target을 만들 때도 현재 neural network를 사용해.
그런데 그 neural network가 계속 업데이트되니까,
target도 계속 바뀌어.
쉽게 말하면:
> 정답으로 삼는 값이 계속 움직이는 상태
야.
그러면 학습이 불안정해질 수 있어.
---
# 10. DQN의 핵심 아이디어 2개
Naive DQN의 문제를 해결하기 위해 DQN은 두 가지를 사용해.
<table header-row="true">
<tr>
<td>문제</td>
<td>해결 방법</td>
</tr>
<tr>
<td>sample들이 서로 연관되어 있음</td>
<td>Experience replay</td>
</tr>
<tr>
<td>target이 계속 바뀜</td>
<td>Target network</td>
</tr>
</table>
이 두 개가 DQN에서 제일 중요해.
---
# 11. Experience Replay
## 의미
Experience replay는 경험 데이터를 buffer에 저장해두고,
학습할 때 무작위로 mini-batch를 뽑아서 사용하는 방법이야.
저장하는 데이터:
(s,a,r,s′)(s,a,r,s')
(s,a,r,s′)
즉:
<table header-row="true">
<tr>
<td>데이터</td>
<td>의미</td>
</tr>
<tr>
<td>sss</td>
<td>현재 state</td>
</tr>
<tr>
<td>aaa</td>
<td>action</td>
</tr>
<tr>
<td>rrr</td>
<td>reward</td>
</tr>
<tr>
<td>s′s's′</td>
<td>next state</td>
</tr>
</table>
---
## 왜 쓰는가?
시간 순서대로 바로 학습하면 sample들이 너무 비슷해.
그래서 buffer에 모아두고 random sampling하면
데이터 간 correlation을 줄일 수 있어.
자료의 핵심:
> Experience replay는 sample 사이의 correlation을 제거하고 bias를 줄인다.
시험용 문장:
> **Experience replay는 transition data를 replay buffer에 저장한 뒤 random sampling하여 학습함으로써 sample correlation을 줄인다.**
---
# 12. Target Network
## 의미
Target network는 target 값을 계산하기 위해 따로 사용하는 neural network야.
DQN에서는 두 개의 network가 있다고 보면 돼.
<table header-row="true">
<tr>
<td>Network</td>
<td>역할</td>
</tr>
<tr>
<td>Current network QθQ_\\thetaQθ</td>
<td>현재 학습 중인 Q-network</td>
</tr>
<tr>
<td>Target network Qθ−Q_\{\\theta\^-\}Qθ−</td>
<td>target 값을 계산하는 network</td>
</tr>
</table>
---
## 왜 쓰는가?
현재 network QθQ_\\thetaQθ는 매번 업데이트돼.
그런데 target도 이 network로 만들면, target이 계속 흔들려.
그래서 target network Qθ−Q_\{\\theta\^-\}Qθ−를 따로 두고,
천천히 업데이트해.
자료의 핵심:
> Target network는 target value가 자주 변하지 않게 해서 stable learning을 가능하게 한다.
시험용 문장:
> **Target network는 target value를 안정적으로 유지하기 위해 current network와 분리해서 사용하는 network이다.**
---
# 13. DQN 알고리즘 흐름
DQN의 전체 흐름은 이렇게 보면 돼.
1. 현재 Q-network와 target network를 초기화한다.
2. ϵ\\epsilonϵ-greedy policy로 action을 선택한다.
3. 환경에서 transition (s,a,r,s′)을 얻는다.
(s,a,r,s′)(s,a,r,s')
4. 이 transition을 replay buffer에 저장한다.
5. buffer에서 mini-batch를 random sampling한다.
6. target network로 target 값을 계산한다.
7. current network가 예측한 Q값과 target의 차이를 loss로 계산한다.
8. gradient descent로 current network를 업데이트한다.
9. 일정 주기마다 target network를 current network와 동기화한다.
---
# 14. Q-function Regression
DQN은 Q값을 맞히는 regression 문제처럼 학습해.
즉, neural network가 예측한 값:
Qθ(s,a)Q_\\theta(s,a)
Qθ(s,a)
과 target 값:
yy
y
의 차이를 줄이는 방향으로 학습해.
쉽게 말하면:
> “이 state-action의 Q값은 이 정도여야 한다”라는 target을 만들고,
network가 그 target에 가까워지도록 학습한다.
그래서 자료에서는 이것을 **Q-function regression**이라고 설명해.
---
# 15. DQN의 장점
자료에서 나온 장점은:
## 15-1. Simple algorithm
DQN은 구조가 비교적 단순해.
기본적으로:
> Q-learning + neural network + replay buffer + target network
라고 보면 돼.
## 15-2. Sample efficiency
DQN은 off-policy이고, experience replay를 사용하기 때문에
한 번 모은 데이터를 여러 번 재사용할 수 있어.
그래서 sample efficiency가 좋다고 설명돼.
---
# 16. DQN의 단점
자료에서 나온 단점은 두 가지야.
## 16-1. Continuous action space에 적용이 어렵다
DQN은 action마다 Q값을 비교해서 max를 골라야 해.
그런데 action이 연속형이면 가능한 action이 무한히 많아져.
그래서 DQN은 continuous action space에 적용하기 어렵다.
이런 경우 policy-based algorithm이 필요할 수 있어.
---
## 16-2. Overestimation 문제
DQN은 Q-learning 기반이라 max 연산을 사용해.
앞에서 배운 것처럼 max 연산은 Q값을 과대평가하는 **maximization bias / overestimation**을 만들 수 있어.
그래서 개선 알고리즘이 등장해.
예:
- Double DQN
- Dueling DQN
시험용 문장:
> **DQN은 Q-learning 기반이므로 overestimation 문제가 발생할 수 있고, 이를 개선하기 위해 Double DQN 등이 제안된다.**
---
# 17. TD Learning과 DQN의 관계
이건 꼭 연결해서 이해해야 해.
<table header-row="true">
<tr>
<td>구분</td>
<td>의미</td>
</tr>
<tr>
<td>Q-learning</td>
<td>table로 Q(s,a)Q(s,a)Q(s,a)를 업데이트</td>
</tr>
<tr>
<td>DQN</td>
<td>neural network로 Qθ(s,a)Q_\\theta(s,a)Qθ(s,a)를 업데이트</td>
</tr>
<tr>
<td>TD Learning</td>
<td>둘 다 Bellman target을 이용한 TD 방식</td>
</tr>
</table>
즉:
> DQN은 Q-learning의 deep learning 버전이다.
앞에서 Q-learning은:
Q(s,a)Q(s,a)
Q(s,a)
를 직접 저장했다면,
DQN은:
Qθ(s,a)Q_\\theta(s,a)
Qθ(s,a)
를 학습해.
---
# 18. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 깊게 안 외워도 돼.
- 전체 update rule 수식
- pseudo-code 전체
- Atari 성능 그래프의 세부 숫자
- CNN 구조의 세부 layer
- Polyak averaging 수식
- 논문 이름 전체
하지만 아래는 반드시 알아야 해.
- DQN = Q-learning + Deep Learning
- RL과 DRL 차이
- value approximation / policy approximation
- Naive DQN의 문제점
- Experience replay
- Target network
- Q-function regression
- DQN의 장단점
- DQN은 discrete action에 적합
- Overestimation 문제
---
# 19. True/False 대비
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>DQN은 Q-learning과 deep learning을 결합한 방법이다.</td>
<td>True</td>
</tr>
<tr>
<td>기존 Q-learning은 Q-table을 사용한다.</td>
<td>True</td>
</tr>
<tr>
<td>DQN은 Q(s,a)Q(s,a)Q(s,a)를 neural network로 근사한다.</td>
<td>True</td>
</tr>
<tr>
<td>DQN은 모든 state-action pair를 table에 저장한다.</td>
<td>False</td>
</tr>
<tr>
<td>DRL은 deep learning과 reinforcement learning을 결합한 것이다.</td>
<td>True</td>
</tr>
<tr>
<td>DQN은 value function approximation에 해당한다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE와 PPO는 policy space approximation에 해당한다.</td>
<td>True</td>
</tr>
<tr>
<td>Naive DQN은 sample correlation과 changing target 문제를 가진다.</td>
<td>True</td>
</tr>
<tr>
<td>Experience replay는 transition을 buffer에 저장하고 random sampling한다.</td>
<td>True</td>
</tr>
<tr>
<td>Experience replay는 sample correlation을 줄이는 역할을 한다.</td>
<td>True</td>
</tr>
<tr>
<td>Target network는 target value를 안정적으로 만들기 위해 사용된다.</td>
<td>True</td>
</tr>
<tr>
<td>DQN은 continuous action space에 적용하기 쉽다.</td>
<td>False</td>
</tr>
<tr>
<td>DQN은 overestimation 문제가 발생할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>Double DQN은 overestimation 문제를 줄이기 위해 등장했다.</td>
<td>True</td>
</tr>
</table>
---
# 최종 암기 요약
이 자료는 아래 문장들만 확실히 잡으면 돼.
> **DQN은 Q-learning과 deep learning을 결합한 방법이다.**
> **기존 Q-learning은 Q-table을 사용하지만, DQN은 neural network로 Q(s,a)Q(s,a)Q(s,a)를 근사한다.**
> **DQN은 큰 state space와 discrete action space에 적합하다.**
> **Naive DQN은 sample correlation과 changing target 문제 때문에 불안정하다.**
> **Experience replay는 transition을 buffer에 저장하고 random sampling하여 sample correlation을 줄인다.**
> **Target network는 target value가 계속 변하는 문제를 줄여 학습을 안정화한다.**
> **DQN은 Q-function regression 문제처럼 target Q-value와 예측 Q-value의 차이를 줄이는 방식으로 학습한다.**
> **DQN은 continuous action space에 적용하기 어렵고, overestimation 문제가 있을 수 있다.**
# 1. discrete가 무슨 뜻이야?
**Discrete = 이산적**이라는 뜻이야.
쉽게 말하면:
> 선택지가 딱딱 끊어져 있는 것
예를 들어 action이 이렇게 정해져 있으면 discrete action이야.
<table header-row="true">
<tr>
<td>Action</td>
<td>의미</td>
</tr>
<tr>
<td>0</td>
<td>주문 안 함</td>
</tr>
<tr>
<td>1</td>
<td>10개 주문</td>
</tr>
<tr>
<td>2</td>
<td>20개 주문</td>
</tr>
<tr>
<td>3</td>
<td>30개 주문</td>
</tr>
</table>
즉, 가능한 action이 **0, 1, 2, 3처럼 정해진 선택지**야.
반대로 **continuous = 연속적**은 값이 무한히 가능하다는 뜻이야.
예를 들어:
> 주문량을 0.1개, 0.2개, 3.75개, 12.836개처럼 아무 실수값으로 선택 가능
이면 continuous action이야.
DQN은 보통:
> **state는 크거나 연속적이어도 가능하지만, action은 discrete인 경우에 적합**
해. 자료에서도 DQN 구조를 **large continuous state space + discrete action space**로 설명하고 있어.
---
# 2. “state가 연속적”이랑 “sample이 연속적”은 다른 말이야?
응, 완전히 달라.
## state가 continuous하다는 뜻
state 값이 실수처럼 다양하게 나올 수 있다는 뜻이야.
예를 들어 재고 문제에서 state가:
> 현재 재고량, 수요 평균, 비용, 시간
처럼 숫자값이면 state space가 커질 수 있어.
이미지 게임에서는 state가 화면 픽셀 전체라서 엄청 커.
이때 Q-table로 모든 state를 저장하기 어렵기 때문에 DQN을 쓰는 거야.
---
## sample이 correlated, sequential하다는 뜻
이건 데이터가 시간 순서대로 이어져 있어서 서로 비슷하다는 뜻이야.
예를 들어 agent가 환경에서 움직이면 데이터가 이렇게 생겨.
s1→s2→s3→s4s_1 \\rightarrow s_2 \\rightarrow s_3 \\rightarrow s_4
s1→s2→s3→s4
그러면 s2s_2s2와 s3s_3s3는 완전히 독립적인 데이터가 아니라,
바로 직전 상태와 다음 상태라서 서로 강하게 관련되어 있어.
이게 **sample correlation**이야.
즉:
<table header-row="true">
<tr>
<td>표현</td>
<td>의미</td>
</tr>
<tr>
<td>continuous state</td>
<td>state 값의 공간이 크거나 연속적임</td>
</tr>
<tr>
<td>correlated samples</td>
<td>경험 데이터들이 시간 순서상 서로 연결되어 있음</td>
</tr>
</table>
Naive DQN의 문제는 **state가 continuous라서**가 아니라,
**학습 데이터 sample들이 시간 순서대로 연관되어 있기 때문**이야. 자료에서도 Naive DQN의 문제로 “samples are correlated”와 “target value is changing”을 따로 설명해.
---
# 3. 그냥 Q-learning에서도 target이 바뀌지 않아?
응, 맞아. **Q-learning에서도 target은 바뀔 수 있어.**
Q-learning target은 보통 이거야.
r+γmax⁡a′Q(s′,a′)r + \\gamma \\max_\{a'\} Q(s', a')
r+γa′maxQ(s′,a′)
여기서 QQQ가 업데이트되면 target도 바뀔 수 있지.
그런데 tabular Q-learning에서는 비교적 문제가 덜해.
왜냐하면 Q-table에서는 특정 Q(s,a)Q(s,a)Q(s,a) 하나를 업데이트하면,
주로 그 칸 하나만 바뀌어.
반면 DQN에서는 neural network parameter θ\\thetaθ를 업데이트해.
그러면 하나의 Q(s,a)Q(s,a)Q(s,a)만 바뀌는 게 아니라,
network가 예측하는 여러 state-action의 Q값이 동시에 바뀔 수 있어.
즉:
<table header-row="true">
<tr>
<td>구분</td>
<td>target 변화 문제</td>
</tr>
<tr>
<td>Tabular Q-learning</td>
<td>특정 table 값 중심으로 업데이트</td>
</tr>
<tr>
<td>Naive DQN</td>
<td>neural network 전체가 바뀌면서 target도 같이 흔들림</td>
</tr>
</table>
그래서 DQN에서는 target network를 따로 둬.
> current network는 계속 학습하고,
target network는 천천히 업데이트해서 target을 안정화한다.
자료에서도 target network는 **non-stationary target**, 즉 계속 바뀌는 target 문제를 줄이기 위한 방법이라고 설명해.
---
# 4. 여기서 sample이 뭐야?
여기서 **sample**은 agent가 환경에서 한 번 행동해서 얻은 경험 데이터야.
보통 하나의 sample은 이렇게 생겨.
(s,a,r,s′)(s, a, r, s')
(s,a,r,s′)
뜻은:
<table header-row="true">
<tr>
<td>기호</td>
<td>의미</td>
</tr>
<tr>
<td>sss</td>
<td>현재 state</td>
</tr>
<tr>
<td>aaa</td>
<td>선택한 action</td>
</tr>
<tr>
<td>rrr</td>
<td>받은 reward</td>
</tr>
<tr>
<td>s′s's′</td>
<td>다음 state</td>
</tr>
</table>
예를 들어 재고 문제라면:
<table header-row="true">
<tr>
<td>구성</td>
<td>예시</td>
</tr>
<tr>
<td>sss</td>
<td>현재 재고 20</td>
</tr>
<tr>
<td>aaa</td>
<td>10개 주문</td>
</tr>
<tr>
<td>rrr</td>
<td>비용이 50 발생해서 reward = -50</td>
</tr>
<tr>
<td>s′s's′</td>
<td>다음 재고 12</td>
</tr>
</table>
이 한 줄이 sample이야.
---
# 5. “action을 선택하고 transition을 얻는다”는 게 무슨 뜻이야?
순서는 이렇게 보면 돼.
현재 상태가 있어.
> s=s =s= 현재 재고 20
agent가 action을 선택해.
> a=a =a= 10개 주문
그 action을 environment에 적용해.
> 수요가 발생함
재고가 변함
비용이 계산됨
그러면 environment가 결과를 줘.
> reward rrr
next state s′s's′
그래서 transition을 얻는다는 건:
> 내가 action을 했더니 환경이 반응해서 (s,a,r,s′)(s,a,r,s')(s,a,r,s′)라는 경험 하나가 생겼다
는 뜻이야.
즉, transition은 미리 만들어놓는 게 아니라,
**agent가 실제로 action을 해보면서 얻는 것**이야.
---
# 6. TD는 transition 없이 가능한 거 아니었어?
아니야. 여기서 중요한 구분이 있어.
TD Learning은 **transition probability 없이 가능**한 거지,
**transition sample 없이 가능한 것**은 아니야.
<table header-row="true">
<tr>
<td>구분</td>
<td>필요 여부</td>
</tr>
<tr>
<td>전체 transition probability (p(s',r</td>
<td>s,a))</td>
</tr>
<tr>
<td>sample transition (s,a,r,s′)(s,a,r,s')(s,a,r,s′)</td>
<td>필요함</td>
</tr>
</table>
즉, TD는 이런 확률표는 몰라도 돼.
> 이 action을 하면 다음 상태 A로 갈 확률 0.3
다음 상태 B로 갈 확률 0.7
하지만 실제 경험은 필요해.
> 이번에 해봤더니 s′s's′로 갔고 reward rrr을 받았다.
자료에서도 TD는 **raw experience, 즉 MDP의 sample transition으로부터 직접 학습한다**고 설명해.
---
# 7. DQN도 TD인데 왜 transition을 먼저 구해?
DQN도 TD 방식이 맞아.
TD 업데이트를 하려면 최소한 이 정보가 필요해.
(s,a,r,s′)(s,a,r,s')
(s,a,r,s′)
왜냐하면 DQN의 target은 이런 식으로 만들어지기 때문이야.
y=r+γmax⁡a′Qθ−(s′,a′)y = r + \\gamma \\max_\{a'\} Q_\{\\theta\^-\}(s',a')
y=r+γa′maxQθ−(s′,a′)
여기서 rrr과 s′s's′가 필요하지?
그래서 먼저 action을 해보고 transition (s,a,r,s′)(s,a,r,s')(s,a,r,s′)을 얻어야 해.
즉:
> transition을 얻는다 → 그 transition으로 TD target을 만든다 → Q-network를 업데이트한다
이 순서야.
---
# 8. Q-function Regression에서 neural network가 예측한 값도 target값이야?
여기서 조심해야 해.
DQN에는 두 값이 있어.
## 1) Prediction, 예측값
현재 network가 예측한 값이야.
Qθ(s,a)Q_\\theta(s,a)
Qθ(s,a)
뜻:
> 현재 network가 “이 state에서 이 action의 Q값은 이 정도야”라고 예측한 값
---
## 2) Target, 목표값
Bellman equation 구조로 만든 학습 목표야.
y=r+γmax⁡a′Qθ−(s′,a′)y = r + \\gamma \\max_\{a'\} Q_\{\\theta\^-\}(s',a')
y=r+γa′maxQθ−(s′,a′)
뜻:
> 이번 경험을 보면, Q(s,a)Q(s,a)Q(s,a)는 이 값에 가까워져야 한다
---
그래서 DQN 학습은:
> prediction Qθ(s,a)Q_\\theta(s,a)Qθ(s,a)와 target yyy의 차이를 줄이는 것
이야.
네가 말한 것처럼:
> 뉴럴 네트워크가 예측한 값과, 우리가 Bellman 방식으로 만든 target값의 차이를 줄이는 방향으로 학습한다
가 맞아.
다만 target값은 사람이 미리 라벨링해둔 정답은 아니야.
Supervised learning에서는 정답 yyy가 데이터에 원래 있어.
DQN에서는 정답이 없으니까,
reward와 다음 state의 Q값을 이용해서 **임시 target**을 만들어.
이걸 **bootstrapped target**이라고 이해하면 돼.
---
# 9. “maximization bias”랑 “overestimation”은 같은 거야?
거의 같은 맥락이야.
TD 자료에서는 **maximization bias**라는 말로 설명했고,
DQN 자료에서는 **overestimation**이라고 표현한 거야.
둘 다 핵심은:
> max 연산 때문에 Q값이 실제보다 크게 추정될 수 있다
는 거야.
Q-learning이나 DQN은 target을 만들 때 이런 걸 써.
max⁡a′Q(s′,a′)\\max_\{a'\} Q(s',a')
a′maxQ(s′,a′)
문제는 Q값이 진짜 값이 아니라 추정값이라는 거야.
추정값들 중에서 가장 큰 값을 고르면,
우연히 높게 추정된 값을 선택할 가능성이 커져.
그래서:
<table header-row="true">
<tr>
<td>표현</td>
<td>의미</td>
</tr>
<tr>
<td>Maximization bias</td>
<td>max 연산 때문에 생기는 편향</td>
</tr>
<tr>
<td>Overestimation</td>
<td>Q값을 실제보다 과대평가하는 현상</td>
</tr>
</table>
즉, 같은 문제를 다른 표현으로 말한 거라고 보면 돼.
---
# 10. 전체 흐름을 다시 정리하면
DQN의 흐름은 이렇게야.
1. 현재 state s를 본다.
ss
2. ϵ\\epsilonϵ-greedy로 action a를 선택한다.
aa
3. environment에 action을 적용한다.
4. reward r, next state s′를 받는다.
rr
s′s'
5. transition (s,a,r,s′)을 replay buffer에 저장한다.
(s,a,r,s′)(s,a,r,s')
6. buffer에서 transition sample들을 random하게 뽑는다.
7. target network로 target y를 만든다.
yy
8. current network의 예측값 Qθ(s,a)와 target y의 차이를 줄인다.
Qθ(s,a)Q_\\theta(s,a)
yy
9. 이 과정을 반복하면서 Q-function을 학습한다.
---
# 진짜 핵심만 압축하면
> **Discrete action**은 선택지가 정해진 action이다. 예를 들어 주문량 0, 10, 20 중 선택하는 것이다.
> **Continuous state**와 **correlated sample**은 다른 말이다. Continuous state는 상태공간이 크거나 연속적이라는 뜻이고, correlated sample은 시간 순서대로 얻은 경험들이 서로 비슷하다는 뜻이다.
> **TD Learning은 transition probability는 몰라도 되지만, sample transition (s,a,r,s′)(s,a,r,s')(s,a,r,s′)은 필요하다.**
> **Sample은 agent가 action을 해서 얻은 하나의 경험 데이터 (s,a,r,s′)(s,a,r,s')(s,a,r,s′)이다.**
> **DQN의 prediction은 Qθ(s,a)Q_\\theta(s,a)Qθ(s,a), target은 r+γmax⁡Qθ−(s′,a′)r+\\gamma \\max Q_\{\\theta\^-\}(s',a')r+γmaxQθ−(s′,a′)이다. DQN은 둘의 차이를 줄이도록 neural network를 학습한다.**
> **Maximization bias와 overestimation은 둘 다 max 연산 때문에 Q값이 과대평가되는 문제를 말한다.**

---

# REINFORCE

---
## 목차
---
## 1. 이 PDF의 큰 주제
한마디로 말하면:
> **Q-value를 먼저 배우는 대신, policy π(a∣s)\\pi(a\|s)π(a∣s) 자체를 직접 학습하는 방법**
이야.
DQN에서는 이런 흐름이었지.
> state → 각 action의 Q값 계산 → Q값이 가장 큰 action 선택
REINFORCE는 조금 달라.
> state → 각 action을 선택할 확률 출력 → 그 확률에 따라 action 선택
즉, REINFORCE는 \*\*“이 상태에서 어떤 action을 몇 % 확률로 선택할까?”\*\*를 직접 학습해.
---
# 2. RL 알고리즘 분류
자료에서는 RL 알고리즘을 크게 세 가지로 나눠.
<table header-row="true">
<tr>
<td>구분</td>
<td>학습하는 것</td>
<td>대표 알고리즘</td>
</tr>
<tr>
<td>**Value-based RL**</td>
<td>V(s)V(s)V(s), Q(s,a)Q(s,a)Q(s,a) 같은 value function</td>
<td>DQN, Double DQN</td>
</tr>
<tr>
<td>**Policy-based RL**</td>
<td>policy (\\pi(a</td>
<td>s)) 자체</td>
</tr>
<tr>
<td>**Actor-Critic**</td>
<td>value function + policy 둘 다</td>
<td>A2C, A3C, DDPG</td>
</tr>
</table>
## 중요 차이
**DQN**은 value-based야.
> 이 action이 얼마나 좋은지 Q(s,a)Q(s,a)Q(s,a)를 학습
**REINFORCE**는 policy-based야.
> 이 action을 선택할 확률 π(a∣s)\\pi(a\|s)π(a∣s)를 직접 학습
**Actor-Critic**은 둘 다 써.
> Actor는 policy를 학습하고, Critic은 value를 학습
---
# 3. Policy parameterization
자료에서 나오는 핵심은 이거야.
πθ(a∣s)\\pi_\\theta(a\|s)
πθ(a∣s)
여기서 θ\\thetaθ는 policy network의 파라미터야.
쉽게 말하면:
> policy를 그냥 표로 저장하는 게 아니라, neural network로 표현한다.
예를 들어 state가 들어오면 neural network가 action별 확률을 출력해.
<table header-row="true">
<tr>
<td>Action</td>
<td>선택 확률</td>
</tr>
<tr>
<td>주문 안 함</td>
<td>0.1</td>
</tr>
<tr>
<td>조금 주문</td>
<td>0.3</td>
</tr>
<tr>
<td>많이 주문</td>
<td>0.6</td>
</tr>
</table>
이 확률분포가 바로 πθ(a∣s)\\pi_\\theta(a\|s)πθ(a∣s)야.
시험용 문장:
> **Policy-based RL은 policy π(a∣s)\\pi(a\|s)π(a∣s)를 파라미터 θ\\thetaθ로 표현하고, 좋은 θ\\thetaθ를 찾는 방법이다.**
---
# 4. Policy Network
REINFORCE는 policy를 **stochastic policy**로 모델링해.
자료 표현:
at∼πθ(at∣st)a_t \\sim \\pi_\\theta(a_t\|s_t)
at∼πθ(at∣st)
뜻은:
> 현재 state sts_tst에서 policy가 정한 확률분포에 따라 action ata_tat를 샘플링한다.
즉, 항상 제일 확률 높은 action만 고르는 게 아니라,
확률적으로 action을 뽑는 거야.
## Continuous action space에서는?
자료에서는 continuous action의 경우 policy network가 평균과 표준편차를 출력할 수 있다고 해.
μ(s),σ(s)\\mu(s), \\sigma(s)
μ(s),σ(s)
그러면 이 분포에서 action을 랜덤하게 샘플링해.
시험용으로는 이 정도면 충분해.
> **REINFORCE는 stochastic policy를 사용하며, action은 policy가 출력한 확률분포에서 샘플링된다.**
---
# 5. REINFORCE의 목표
REINFORCE의 목표는:
> **기대 누적 reward를 최대화하는 policy parameter θ\\thetaθ를 찾는 것**
자료에서는 목적함수를 이렇게 둬.
J(θ)J(\\theta)
J(θ)
의미는:
> policy πθ\\pi_\\thetaπθ를 따라 episode를 진행했을 때 기대되는 총 reward
즉, REINFORCE는:
> J(θ)J(\\theta)J(θ)가 커지도록 θ\\thetaθ를 업데이트하는 알고리즘
이야.
---
# 6. Gradient ascent
DQN은 loss를 줄이는 방향으로 학습했지.
REINFORCE는 reward를 키우고 싶으니까 **gradient ascent**를 써.
θ←θ+α∇θJ(θ)\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)
θ←θ+α∇θJ(θ)
뜻은:
> reward가 증가하는 방향으로 policy parameter를 조금 이동시킨다.
여기서 α\\alphaα는 step size, 즉 learning rate야.
시험용 문장:
> **REINFORCE는 기대 reward J(θ)J(\\theta)J(θ)를 최대화하기 위해 gradient ascent를 사용한다.**
---
# 7. Policy Gradient의 직관
수식은 복잡하지만 핵심은 엄청 간단해.
REINFORCE는 episode를 하나 실행해보고 이렇게 판단해.
## 어떤 action을 했는데 결과가 좋았다면?
> 그 action을 다음에 더 자주 선택하도록 확률을 올린다.
## 어떤 action을 했는데 결과가 나빴다면?
> 그 action을 다음에 덜 선택하도록 확률을 낮춘다.
즉, REINFORCE는:
> **좋은 return을 만든 action의 선택 확률을 증가시키는 방법**
이야.
---
# 8. REINFORCE 업데이트 의미
자료의 기본 업데이트는 이 형태야.
θ←θ+α∇θlog⁡πθ(st,at)⋅Gt\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta \\log \\pi_\\theta(s_t,a_t) \\cdot G_t
θ←θ+α∇θlogπθ(st,at)⋅Gt
여기서 각 항의 의미만 알면 돼.
<table header-row="true">
<tr>
<td>항</td>
<td>의미</td>
</tr>
<tr>
<td>θ\\thetaθ</td>
<td>policy network parameter</td>
</tr>
<tr>
<td>(\\pi_\\theta(a_t</td>
<td>s_t))</td>
</tr>
<tr>
<td>(\\log \\pi_\\theta(a_t</td>
<td>s_t))</td>
</tr>
<tr>
<td>GtG_tGt</td>
<td>그 시점 이후 받은 return</td>
</tr>
<tr>
<td>α\\alphaα</td>
<td>learning rate</td>
</tr>
</table>
핵심은:
> GtG_tGt(return)가 크면, 그 action의 확률을 높이는 방향으로 업데이트한다.
즉:
> 결과가 좋았던 행동은 더 자주 하게 만들고,
결과가 나빴던 행동은 덜 하게 만든다.
---
# 9. GtG_tGt, Return after t
REINFORCE는 Monte Carlo 방식이야.
즉, 한 episode를 끝까지 해보고 나서 return을 계산해.
Gt=∑k=t+1TrkG_t = \\sum_\{k=t+1\}\^\{T\} r_k
Gt=k=t+1∑Trk
뜻:
> 시점 ttt 이후부터 episode 끝까지 받은 reward의 합
DQN이나 TD Learning은 한 step 후의 reward와 next state를 이용해서 바로 업데이트했지.
REINFORCE는 다르게:
> episode를 끝까지 진행한 뒤, 실제 받은 return GtG_tGt를 이용해 업데이트
해.
시험용 문장:
> **REINFORCE는 Monte Carlo 기반 방법으로, episode가 끝난 뒤 return을 계산해 policy를 업데이트한다.**
---
# 10. REINFORCE와 DQN 차이
<table header-row="true">
<tr>
<td>구분</td>
<td>DQN</td>
<td>REINFORCE</td>
</tr>
<tr>
<td>알고리즘 종류</td>
<td>Value-based</td>
<td>Policy-based</td>
</tr>
<tr>
<td>학습 대상</td>
<td>Q(s,a)Q(s,a)Q(s,a)</td>
<td>(\\pi_\\theta(a</td>
</tr>
<tr>
<td>action 선택</td>
<td>Q값이 큰 action 선택</td>
<td>policy 확률분포에서 action 샘플링</td>
</tr>
<tr>
<td>업데이트 기준</td>
<td>TD target</td>
<td>Monte Carlo return</td>
</tr>
<tr>
<td>대표 특징</td>
<td>discrete action에 적합</td>
<td>policy를 직접 학습</td>
</tr>
</table>
핵심:
> DQN은 “각 action의 점수”를 배우고, REINFORCE는 “action 선택 확률”을 배운다.
---
# 11. On-policy 특징
REINFORCE는 **on-policy**야.
즉:
> 현재 policy로 episode를 만들고, 그 episode로 현재 policy를 업데이트해야 한다.
자료에서도 단점으로:
> On-policy: must use the most recent policy
라고 설명해.
이 말은:
> 예전 policy로 만든 데이터를 계속 재사용하기 어렵다
는 뜻이야.
그래서 sample이 많이 필요해질 수 있어.
시험용 문장:
> **REINFORCE는 on-policy 방법이므로, 현재 policy로 생성한 episode를 사용해 업데이트한다.**
---
# 12. REINFORCE의 장점
자료에서 나온 장점은 두 가지야.
<table header-row="true">
<tr>
<td>장점</td>
<td>의미</td>
</tr>
<tr>
<td>Simple</td>
<td>알고리즘 구조가 비교적 단순</td>
</tr>
<tr>
<td>Unbiased gradient</td>
<td>Monte Carlo return을 사용하므로 gradient 추정이 편향되지 않음</td>
</tr>
</table>
즉:
> 실제 episode 결과를 이용하므로 이론적으로는 unbiased한 gradient 추정이 가능하다.
---
# 13. REINFORCE의 단점
중요한 단점은:
> **gradient variance가 크다**
쉽게 말하면:
> episode마다 결과가 너무 들쭉날쭉해서 학습이 불안정할 수 있다.
예를 들어 같은 policy로 해도 어떤 episode는 운 좋게 reward가 높고,
어떤 episode는 운 나쁘게 reward가 낮을 수 있어.
그러면 업데이트 방향이 많이 흔들려.
또 다른 단점:
> on-policy라서 sample을 많이 필요로 한다.
시험용 문장:
> **REINFORCE는 단순하고 unbiased하지만, gradient variance가 크고 sample이 많이 필요할 수 있다.**
---
# 14. Baseline은 왜 쓰는가?
Baseline은 variance를 줄이기 위해 사용해.
기본 REINFORCE는 GtG_tGt를 그대로 써.
그런데 GtG_tGt가 너무 들쭉날쭉하면 업데이트가 불안정해.
그래서 baseline bbb를 빼.
Gt−bG_t - b
Gt−b
중요한 점:
> baseline을 빼도 expectation은 바뀌지 않지만 variance를 줄일 수 있다.
쉽게 말하면:
> 절대 점수 GtG_tGt를 보는 대신, 평균보다 잘했는지 못했는지를 보는 것
이야.
예를 들어 평균 return이 100이라고 해보자.
<table header-row="true">
<tr>
<td>Return GtG_tGt</td>
<td>Baseline bbb</td>
<td>Gt−bG_t-bGt−b</td>
<td>의미</td>
</tr>
<tr>
<td>150</td>
<td>100</td>
<td>50</td>
<td>평균보다 잘함</td>
</tr>
<tr>
<td>80</td>
<td>100</td>
<td>-20</td>
<td>평균보다 못함</td>
</tr>
</table>
이렇게 하면 업데이트가 더 안정적이야.
시험용 문장:
> **Baseline은 policy gradient의 expectation을 바꾸지 않으면서 variance를 줄이기 위해 사용된다.**
---
# 15. 어떤 baseline을 쓰나?
자료에서는 여러 baseline을 말해.
## 15-1. State-value function baseline
b(s)=vπ(s)b(s) = v\^\\pi(s)
b(s)=vπ(s)
뜻:
> 해당 state에서 기대되는 평균 return을 baseline으로 사용
그러면:
Gt−vπ(st)G_t - v\^\\pi(s_t)
Gt−vπ(st)
가 돼.
이 값은 나중에 **advantage**와 연결돼.
의미는:
> 이 action이 이 state에서 평균보다 얼마나 좋았는가?
---
## 15-2. Actor-Critic과 연결
자료에서 중요한 문장:
> If we train v(st)v(s_t)v(st), this structure becomes an actor-critic.
즉, baseline으로 value function을 따로 학습하면 Actor-Critic 구조가 돼.
<table header-row="true">
<tr>
<td>역할</td>
<td>의미</td>
</tr>
<tr>
<td>Actor</td>
<td>policy (\\pi_\\theta(a</td>
</tr>
<tr>
<td>Critic</td>
<td>value function v(s)v(s)v(s) 학습</td>
</tr>
</table>
그래서 REINFORCE with baseline은 Actor-Critic으로 이어지는 중간 단계처럼 이해하면 좋아.
---
# 16. Other Baselines
자료에는 세 가지 baseline 예시가 나와.
<table header-row="true">
<tr>
<td>Baseline</td>
<td>의미</td>
<td>특징</td>
</tr>
<tr>
<td>Stochastic rollout</td>
<td>여러 trajectory를 샘플링해서 평균 return 사용</td>
<td>unbiased지만 비용 큼</td>
</tr>
<tr>
<td>Greedy rollout</td>
<td>stochastic sampling 없이 greedy하게 rollout</td>
<td>비용 낮지만 biased</td>
</tr>
<tr>
<td>Actor-Critic</td>
<td>neural network로 value function 근사</td>
<td>sample efficient하지만 추가 학습 필요</td>
</tr>
</table>
시험에서는 이름과 특징 정도만 알면 돼.
---
# 17. REINFORCE와 TD Learning 차이
<table header-row="true">
<tr>
<td>구분</td>
<td>TD Learning / Q-learning</td>
<td>REINFORCE</td>
</tr>
<tr>
<td>업데이트 시점</td>
<td>매 step마다 가능</td>
<td>episode 끝난 뒤</td>
</tr>
<tr>
<td>학습 대상</td>
<td>Q(s,a)Q(s,a)Q(s,a)</td>
<td>policy (\\pi(a</td>
</tr>
<tr>
<td>target</td>
<td>reward + next value</td>
<td>실제 return GtG_tGt</td>
</tr>
<tr>
<td>방식</td>
<td>TD 기반</td>
<td>Monte Carlo 기반</td>
</tr>
<tr>
<td>대표 문제</td>
<td>value를 학습</td>
<td>policy를 직접 학습</td>
</tr>
</table>
핵심:
> TD는 한 step씩 bootstrap하고, REINFORCE는 episode 전체 return을 이용한다.
---
# 18. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 깊게 안 외워도 돼.
- policy gradient 유도 과정 전체
- 적분 수식
- log trick 세부 증명
- baseline variance 수식 전체
- batch gradient 식 전체
- rollout baseline 수식 세부
하지만 아래는 꼭 알아야 해.
- Value-based / Policy-based / Actor-Critic 구분
- REINFORCE는 policy-based
- πθ(a∣s)\\pi_\\theta(a\|s)πθ(a∣s) 의미
- stochastic policy
- gradient ascent
- Monte Carlo return Gt
GtG_t
- REINFORCE의 장단점
- baseline의 역할
- baseline과 Actor-Critic 연결
---
# 19. True/False 대비
<table header-row="true">
<tr>
<td>문장</td>
<td>답</td>
</tr>
<tr>
<td>REINFORCE는 policy-based RL 알고리즘이다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 Q(s,a)Q(s,a)Q(s,a) table을 직접 학습하는 알고리즘이다.</td>
<td>False</td>
</tr>
<tr>
<td>Policy-based RL은 policy (\\pi(a</td>
<td>s))를 직접 학습한다.</td>
</tr>
<tr>
<td>REINFORCE는 stochastic policy를 사용할 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 action을 policy 확률분포에서 샘플링한다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 Monte Carlo 기반 학습 방법이다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 episode 단위로 업데이트한다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 gradient ascent를 사용해 J(θ)J(\\theta)J(θ)를 최대화한다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 on-policy 방법이다.</td>
<td>True</td>
</tr>
<tr>
<td>REINFORCE는 variance가 낮아서 항상 안정적이다.</td>
<td>False</td>
</tr>
<tr>
<td>Baseline은 variance를 줄이기 위해 사용된다.</td>
<td>True</td>
</tr>
<tr>
<td>Baseline을 빼면 expectation은 유지하면서 variance를 줄일 수 있다.</td>
<td>True</td>
</tr>
<tr>
<td>b(s)=v(s)b(s)=v(s)b(s)=v(s)를 baseline으로 학습하면 Actor-Critic 구조와 연결된다.</td>
<td>True</td>
</tr>
</table>
---
# 최종 암기 요약
이 자료는 아래 문장들만 확실히 잡으면 돼.
> **REINFORCE는 policy πθ(a∣s)\\pi_\\theta(a\|s)πθ(a∣s)를 직접 학습하는 policy-based RL 알고리즘이다.**
> **DQN은 Q-value를 학습하지만, REINFORCE는 action 선택 확률을 학습한다.**
> **REINFORCE는 stochastic policy를 사용하며, action은 policy가 출력한 확률분포에서 샘플링된다.**
> **REINFORCE의 목표는 기대 누적 reward J(θ)J(\\theta)J(θ)를 최대화하는 policy parameter θ\\thetaθ를 찾는 것이다.**
> **REINFORCE는 gradient ascent를 사용해 reward가 증가하는 방향으로 policy parameter를 업데이트한다.**
> **REINFORCE는 Monte Carlo 기반 방법이라 episode가 끝난 뒤 return GtG_tGt를 이용해 업데이트한다.**
> **REINFORCE는 단순하고 unbiased하지만, gradient variance가 크고 on-policy라 sample이 많이 필요하다.**
> **Baseline은 expectation을 바꾸지 않으면서 variance를 줄이기 위해 사용된다.**
> **State-value function을 baseline으로 학습하면 Actor-Critic 구조와 연결된다.**

---

# Actor-Critic

---
## 목차
---
## 1. 이 PDF의 큰 주제
한마디로 말하면:
> **REINFORCE처럼 policy를 직접 학습하되, value function을 같이 학습해서 더 안정적으로 업데이트하는 방법**
이야.
REINFORCE는 episode가 끝난 뒤 실제 return GtG_tGt를 사용했지.
그런데 GtG_tGt는 episode 전체 결과라서 변동이 커.
그래서 Actor-Critic은 이렇게 해.
```plain text
Actor:
어떤 action을 할지 결정하는 policy를 학습

Critic:
그 action이 기대보다 좋았는지 나빴는지 평가하는 value function을 학습
```
즉:
> **Actor-Critic = Policy-based RL + Value function**
---
# 2. RL 알고리즘 분류에서 위치
이 자료에서도 RL 알고리즘을 세 가지로 나눠.
```plain text
Value-based RL
- V(s) 또는 Q(s,a)를 학습
- 예: DQN, Double DQN

Policy-based RL
- policy π(a|s)를 직접 학습
- 예: REINFORCE, PPO

Actor-Critic
- policy와 value function을 동시에 학습
- 예: A2C, A3C, DDPG
```
그러니까 Actor-Critic은 **value-based와 policy-based의 중간**이라고 보면 돼.
자료 그림에서도 Actor-Critic은 **Value Function 영역과 Policy 영역이 겹치는 부분**에 있어.
---
# 3. REINFORCE에서 Actor-Critic으로 넘어가는 이유
REINFORCE는 이렇게 업데이트했어.
θ←θ+α∇θlog⁡πθ(at∣st)Gt\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta \\log \\pi_\\theta(a_t\|s_t)G_t
θ←θ+α∇θlogπθ(at∣st)Gt
뜻은:
> 어떤 action을 했고, 그 뒤 return GtG_tGt가 좋으면 그 action의 확률을 올린다.
그런데 문제는 GtG_tGt야.
GtG_tGt는 episode 끝까지의 reward 합이라서 너무 흔들릴 수 있어.
그래서 REINFORCE with baseline에서는 이렇게 했지.
Gt−b(st)G_t - b(s_t)
Gt−b(st)
여기서 baseline을 **state-value functio**n으로 두면:
b(st)=Vπ(st)b(s_t) = V\^\\pi(s_t)
b(st)=Vπ(st)
이게 Actor-Critic의 출발점이야.
즉:
> **baseline을 value function으로 학습하면 Actor-Critic 구조가 된다.**
---
# 4. Actor와 Critic의 역할
## Actor
Actor는 policy를 학습해.
πθ(a∣s)\\pi_\\theta(a\|s)
πθ(a∣s)
뜻:
> state sss에서 action aaa를 선택할 확률
Actor는 실제로 action을 고르는 역할을 해.
예를 들어 재고 문제라면:
```plain text
state = 현재 재고 20

Actor 출력:
주문 0개  확률 0.1
주문 10개 확률 0.6
주문 20개 확률 0.3
```
Actor는 이 확률에 따라 action을 선택해.
---
## Critic
Critic은 value function을 학습해.
Vϕ(s)V_\\phi(s)
Vϕ(s)
뜻:
> state sss에서 앞으로 기대되는 return
Critic은 Actor가 고른 action이 좋았는지 평가해.
즉:
```plain text
Actor:
이 action을 하자.

Critic:
그 action은 기대보다 좋았는지 나빴는지 평가해줄게.
```
자료 표현으로는:
> Actor decides which action to take.
Critic evaluates how good the selected action was.
Actor is updated using the critic’s evaluation.
---
# 5. Advantage Function이 뭐야?
Actor-Critic에서 매우 중요한 개념이 **advantage**야.
Advantage는 쉽게 말하면:
> **그 action이**** 평균보다 ****얼마나 좋았는가**
를 나타내.
수식은:
Aπ(st,at)=Qπ(st,at)−Vπ(st)A\^\\pi(s_t,a_t) = Q\^\\pi(s_t,a_t) - V\^\\pi(s_t)
Aπ(st,at)=Qπ(st,at)−Vπ(st)
뜻은:
```plain text
Q(s,a):
이 state에서 이 action을 했을 때의 가치

V(s):
이 state에서 평균적으로 기대되는 가치

A(s,a):
이 action이 평균보다 얼마나 좋은지
```
예를 들어 어떤 state에서 평균적으로 기대되는 return이 100이라고 해보자.
```plain text
이번 action 이후 return = 130
평균 기대 return = 100
Advantage = +30
```
그러면 이 action은 기대보다 좋았던 거야.
반대로:
```plain text
이번 action 이후 return = 70
평균 기대 return = 100
Advantage = -30
```
그러면 이 action은 기대보다 나빴던 거야.
---
# 6. Advantage가 양수/음수일 때 의미
```plain text
A_t > 0
→ action이 기대보다 좋았다.
→ Actor는 그 action의 선택 확률을 높인다.

A_t < 0
→ action이 기대보다 나빴다.
→ Actor는 그 action의 선택 확률을 낮춘다.
```
이게 Actor update의 핵심이야.
Actor는 단순히 reward가 컸는지만 보는 게 아니라,
> **기대보다 잘했는가?**
를 보고 policy를 수정해.
---
# 7. Actor Update
Actor는 advantage를 이용해서 policy를 업데이트해.
θ←θ+αθ∇θlog⁡πθ(at∣st)At\\theta \\leftarrow \\theta + \\alpha_\\theta \\nabla_\\theta \\log \\pi_\\theta(a_t\|s_t) A_t
θ←θ+αθ∇θlogπθ(at∣st)At
이 식의 의미는 이거야.
```plain text
A_t가 양수이면:
방금 선택한 action의 확률을 올린다.

A_t가 음수이면:
방금 선택한 action의 확률을 내린다.
```
즉, Actor의 역할은:
> **기대보다 좋은 action을 더 자주 선택하도록 policy를 바꾸는 것**
이야.
---
# 8. Critic Update
Critic은 value function을 더 정확하게 만들려고 학습해.
Critic의 목표는:
> Vϕ(s)V_\\phi(s)Vϕ(s)가 실제 return 또는 TD target에 가까워지도록 학습하는 것
Monte Carlo 방식에서는 critic loss가 이렇게 돼.
Lcritic=(Gt−Vϕ(st))2L_\{\\text\{critic\}\} = (G_t - V_\\phi(s_t))\^2
Lcritic=(Gt−Vϕ(st))2
뜻은:
```plain text
실제로 받은 return G_t
-
critic이 예측한 value V(s_t)

이 차이를 줄이자.
```
즉, Critic은 **state value를 잘 예측하도록 학습**해.
---
# 9. Monte Carlo Actor-Critic
처음에는 REINFORCE처럼 episode가 끝난 뒤 return GtG_tGt를 계산해서 Actor와 Critic을 업데이트할 수 있어.
흐름은 이거야.
```plain text
1. Actor가 현재 policy로 episode를 끝까지 실행한다.
2. 각 step의 return G_t를 계산한다.
3. Critic이 V(s_t)를 예측한다.
4. Advantage = G_t - V(s_t)를 계산한다.
5. Actor는 advantage로 policy를 업데이트한다.
6. Critic은 G_t에 맞게 V(s_t)를 업데이트한다.
```
이건 REINFORCE with baseline과 매우 비슷해.
차이는 baseline V(s)V(s)V(s)를 **Critic이 직접 학습한다**는 점이야.
---
# 10. 그런데 왜 TD Actor-Critic으로 넘어가?
Monte Carlo 방식은 여전히 episode가 끝날 때까지 기다려야 해.
그래서 자료에서는 TD 방식으로 넘어가.
TD target은:
rt+1+γVϕ(st+1)r_\{t+1\} + \\gamma V_\\phi(s_\{t+1\})
rt+1+γVϕ(st+1)
뜻:
> 지금 받은 reward + 다음 state의 예상 value
그래서 TD error는:
δt=rt+1+γVϕ(st+1)−Vϕ(st)\\delta_t = r_\{t+1\} + \\gamma V_\\phi(s_\{t+1\}) - V_\\phi(s_t)
δt=rt+1+γVϕ(st+1)−Vϕ(st)
이 δt\\delta_tδt를 advantage처럼 사용해.
즉:
At≈δtA_t \\approx \\delta_t
At≈δt
---
# 11. TD error가 advantage 역할을 한다는 게 무슨 뜻이야?
TD error는 이렇게 해석할 수 있어.
```plain text
예상했던 현재 state의 가치:
V(s_t)

실제로 한 step 가보니 보인 가치:
r + γV(s_{t+1})

차이:
δ_t = r + γV(s_{t+1}) - V(s_t)
```
만약 δt\>0\\delta_t \> 0δt\>0이면?
```plain text
생각보다 결과가 좋았다.
→ 이 action의 확률을 올린다.
```
만약 δt\<0\\delta_t \< 0δt\<0이면?
```plain text
생각보다 결과가 나빴다.
→ 이 action의 확률을 낮춘다.
```
그래서 TD Actor-Critic에서는 δt\\delta_tδt를 advantage estimate로 사용해.
---
# 12. TD Actor-Critic의 장점
TD Actor-Critic은 episode 끝까지 기다리지 않아도 돼.
한 step만 지나도 업데이트 가능해.
```plain text
state s_t 관찰
Actor가 action a_t 선택
reward r_{t+1}, next state s_{t+1} 관찰
TD error δ_t 계산
Actor 업데이트
Critic 업데이트
```
즉:
> **TD Actor-Critic은 매 step마다 업데이트할 수 있다.**
이 점 때문에 REINFORCE보다 sample efficiency가 좋아질 수 있어.
---
# 13. Loss Function 관점
자료에서는 Actor-Critic을 loss function으로도 설명해.
## Critic loss
Lcritic=(rt+1+γVϕ(st+1)−Vϕ(st))2L_\{\\text\{critic\}\} = (r_\{t+1\} + \\gamma V_\\phi(s_\{t+1\}) - V_\\phi(s_t))\^2
Lcritic=(rt+1+γVϕ(st+1)−Vϕ(st))2
뜻:
> Critic의 value 예측이 TD target과 가까워지도록 학습
---
## Actor loss
Lactor=−log⁡πθ(at∣st)δtL_\{\\text\{actor\}\} = -\\log \\pi_\\theta(a_t\|s_t)\\delta_t
Lactor=−logπθ(at∣st)δt
뜻:
> δt\\delta_tδt가 양수이면 해당 action의 확률을 높이고,
δt\\delta_tδt가 음수이면 해당 action의 확률을 낮추도록 학습
---
# 14. Entropy Bonus
Entropy bonus는 exploration을 유지하기 위한 장치야.
policy가 너무 빨리 deterministic해지면 문제가 생겨.
예를 들어 학습 초반에 우연히 어떤 action이 좋아 보였다고 해보자.
그러면 policy가 너무 빨리 이렇게 될 수 있어.
```plain text
주문 0개  확률 0.99
주문 10개 확률 0.01
주문 20개 확률 0.00
```
이렇게 되면 다른 action을 거의 시도하지 않게 돼.
그래서 entropy bonus를 넣어.
Entropy가 높다는 건:
> action 선택이 더 랜덤하고 다양하다
는 뜻이야.
즉:
> **Entropy bonus는 policy가 너무 빨리 한 action에 고정되는 것을 막고 exploration을 장려한다.**
---
# 15. Actor-Critic vs REINFORCE
자료 기준으로 차이를 정리하면 이렇게 보면 돼.
```plain text
REINFORCE
- update target: full return G_t
- update timing: episode-by-episode
- variance: high
- bias: low
- value function: optional baseline
- sample efficiency: low

Actor-Critic
- update target: TD error δ_t
- update timing: step-by-step 또는 rollout
- variance: lower
- bias: 있을 수 있음
- value function: essential critic
- sample efficiency: higher
```
핵심 차이는 이거야.
> **REINFORCE는 실제 return GtG_tGt를 사용하고, Actor-Critic은 Critic이 만든 TD error δt\\delta_tδt를 사용한다.**
---
# 16. Actor-Critic이 더 잘 작동하는 이유
자료 마지막 부분의 핵심이야.
Actor-Critic이 REINFORCE보다 나을 수 있는 이유는:
```plain text
1. REINFORCE는 episode 끝까지 기다린다.
2. Actor-Critic은 episode가 끝나기 전에 업데이트할 수 있다.
3. Critic이 더 안정적인 학습 신호를 제공한다.
4. Actor는 그 신호를 이용해 policy를 개선한다.
5. 그래서 variance가 낮아지고, 학습이 빨라지고, sample efficiency가 좋아질 수 있다.
```
---
# 17. 이 자료에서 안 외워도 되는 것
quiz 형식이면 아래는 깊게 안 외워도 돼.
```plain text
- Actor loss, critic loss의 세부 유도
- gradient 식 전체
- batch version pseudo-code 전체
- entropy 수식 전체
- TD Actor-Critic update rule 전체 암기
```
대신 아래는 꼭 알아야 해.
```plain text
- Actor와 Critic의 역할
- Actor-Critic이 REINFORCE에서 어떻게 이어지는지
- Advantage function의 의미
- TD error가 advantage estimate로 사용된다는 점
- Actor update의 직관
- Critic update의 직관
- Entropy bonus의 역할
- Actor-Critic과 REINFORCE 차이
```
---
# 18. True/False 대비
```plain text
Actor-Critic은 policy와 value function을 동시에 학습한다.
→ True

Actor는 value function을 학습하는 역할이다.
→ False
Actor는 policy를 학습한다.

Critic은 선택된 action이 얼마나 좋았는지 평가한다.
→ True

Advantage는 action이 평균보다 얼마나 좋은지를 나타낸다.
→ True

Advantage가 양수이면 해당 action의 확률을 높이는 방향으로 업데이트한다.
→ True

REINFORCE는 반드시 critic을 사용한다.
→ False

Actor-Critic에서는 critic이 value function을 학습한다.
→ True

TD Actor-Critic은 episode가 끝나기 전에도 업데이트할 수 있다.
→ True

Entropy bonus는 exploration을 줄이기 위해 사용된다.
→ False
Entropy bonus는 exploration을 장려하기 위해 사용된다.

Actor-Critic은 REINFORCE보다 variance를 줄이고 sample efficiency를 높일 수 있다.
→ True
```
---
# 최종 암기 요약
```plain text
Actor-Critic은 policy를 학습하는 Actor와 value function을 학습하는 Critic을 동시에 사용하는 강화학습 방법이다.

Actor는 state를 보고 action을 선택하는 policy πθ(a|s)를 학습한다.

Critic은 state value Vϕ(s)를 학습해서 Actor가 선택한 action이 좋았는지 평가한다.

Advantage는 어떤 action이 평균보다 얼마나 좋은지를 나타낸다.

Advantage가 양수이면 그 action의 확률을 올리고, 음수이면 그 action의 확률을 낮춘다.

TD Actor-Critic에서는 TD error δt를 advantage estimate로 사용한다.

REINFORCE는 episode 끝까지 기다려 Gt로 업데이트하지만, Actor-Critic은 step마다 업데이트할 수 있다.

Entropy bonus는 policy가 너무 빨리 한 action에 고정되는 것을 막고 exploration을 장려한다.

Actor-Critic은 REINFORCE보다 variance를 줄이고 sample efficiency를 높일 수 있지만, bootstrapping 때문에 bias가 생길 수 있다
```
