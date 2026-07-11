# Robot Arm 2D — Trajectory Tracking với Deep RL

Báo cáo đồ án môn Deep Reinforcement Learning.

## 1. Mục tiêu

Huấn luyện tay máy 2 khớp phẳng (2-link planar arm, l1 = l2 = 1.0) học policy
điều khiển để end-effector (ee) bám theo target: điểm cố định → đường tròn →
hình số 8 → spline tự vẽ, bằng RL continuous control (SAC, PPO — Stable-Baselines3).

## 2. Môi trường (custom Gymnasium env)

**Observation** (Box, float32, shape (10,)):
`[cos θ1, sin θ1, cos θ2, sin θ2, θ1_dot, θ2_dot, ex, ey, ex−px, ey−py]`
— dùng cos/sin thay θ thô để tránh wrap-around 2π; 2 phần tử cuối là error
vector (target − ee) giúp policy bám được target di động.

**Action** (Box, shape (2,), [−1, 1]): kinematic control, Δθ_i = action_i × 0.1 rad.

**Reward** = tổng các term (hệ số trong config, log riêng từng term vào `info`):

| Term | Công thức | Hệ số mặc định |
|---|---|---|
| dist | −‖ee − target‖ | 1 (luôn bật) |
| effort | −w_e·‖action‖ | w_e = 0.01 |
| jerk | −w_j·‖θ_dot‖ | w_j = 0.001 |
| bonus | +b nếu dist < 0.05 | b = 1.0 |

**Episode**: 200 bước (dt = 0.05 s → 10 s), `truncated` khi hết bước,
`terminated` luôn False (tracking là task liên tục).

**Quỹ đạo target** (`robot_arm/trajectories.py`, param theo thời gian t):
- `circle`: tâm (1.0, 0), bán kính 0.5, chu kỳ 10 s (1 vòng/episode).
- `figure8`: Lissajous 1:2 — [a·sin θ, b·sin 2θ], a = 0.6, b = 0.35.
- `spline`: Catmull-Rom khép kín qua 5 control points tự vẽ.
- Mỗi episode random offset t0 để đa dạng điểm xuất phát.

## 3. Setup thí nghiệm

- SAC và PPO từ SB3 2.9.0, MlpPolicy (mặc định 256×256), seed 42, 100k timesteps.
- Train bằng job SLURM trên partition CPU (net nhỏ nên CPU nhanh hơn GPU ~3×:
  ~110 fps vs 38 fps trên A100 — overhead chuyển dữ liệu CPU↔GPU lấn át).
- Mỗi run lưu riêng: `outputs/runs/<run>/` (config, checkpoints, tensorboard, eval).
- Đánh giá bằng `eval.py`: 20 episodes deterministic — mean/max tracking error,
  success rate (% bước dist < 0.05), mean ‖action‖ (smoothness), video overlay.

## 4. Kết quả

### 4.1 Reaching điểm cố định (SAC, 100k steps)

Mean error 0.120, success rate 86.9%. Reward tăng từ −427 (random) lên ~−27;
mean distance giảm 2.05 → ~0.1 sau ~30k steps. Max error 3.2 là khoảng cách
đầu episode (target random khắp workspace), không phải lỗi hội tụ.

### 4.2 SAC vs PPO (task circle, cùng 100k steps, seed 42)

| Metric | SAC | PPO |
|---|---|---|
| Mean tracking error | **0.067** | 0.072 |
| Success rate | **90.6%** | 90.4% |
| Mean ‖action‖ | 0.283 | 0.279 |
| Wall-clock train | 16 m 02 s | **1 m 24 s** |

Learning curve: `outputs/learning_curves_sac_vs_ppo.png`. SAC hội tụ sau
~25–30k steps, PPO cần ~50–60k — SAC sample-efficient hơn (off-policy, replay
buffer) đúng lý thuyết; nhưng wall-clock PPO nhanh hơn ~11× vì chỉ update mỗi
2048 steps thay vì mỗi step. Chất lượng cuối cùng tương đương.

### 4.3 Ba loại quỹ đạo (SAC, 100k steps)

| Quỹ đạo | Mean error | Max error | Success rate | Mean ‖action‖ |
|---|---|---|---|---|
| Circle | **0.067** | 1.49 | **90.6%** | 0.28 |
| Figure-8 | 0.082 | 1.56 | 89.9% | 0.35 |
| Spline | 0.101 | 2.60 | 87.5% | 0.39 |

Quỹ đạo càng gắt (đổi hướng nhiều, cong không đều) thì error và effort càng
tăng. Cả 3 hội tụ sau ~25–30k steps
(`outputs/learning_curves_trajectories.png`); video overlay trong
`outputs/runs/<run>/eval/overlay.gif` cho thấy ee bám khít target sau pha
tiếp cận đầu episode.

### 4.4 Ablation reward terms (base: SAC circle)

Bật/tắt từng term phụ, giữ nguyên mọi thứ khác (seed 42, 100k steps).
So sánh bằng tracking error — episode reward KHÔNG so sánh được giữa các
biến thể vì thang reward khác nhau (bonus +1/bước đẩy reward lên ~+160).

| Biến thể | Mean error | Success rate | Mean ‖action‖ |
|---|---|---|---|
| base (chỉ −dist) | 0.067 | 90.6% | 0.283 |
| + effort (w=0.01) | 0.071 | 90.6% | 0.283 |
| + jerk (w=0.001) | 0.069 | 90.6% | 0.282 |
| + bonus (b=1.0) | 0.071 | 90.6% | 0.282 |
| full (cả 3 term) | 0.070 | 90.6% | 0.282 |

**Kết luận ablation**: với hệ số đề xuất, các term phụ không thay đổi đáng kể
cả tracking error lẫn smoothness (learning curve mean_distance của 5 biến thể
trùng khít — `outputs/learning_curves_ablation.png`). Lý do: (1) term −dist
lớn hơn penalty 1–2 bậc độ lớn nên vẫn chi phối gradient; (2) policy học từ
−dist vốn đã mượt — kinematic control với Δθ ≤ 0.1 rad tự giới hạn action;
(3) bonus chỉ dịch chuyển thang reward chứ không đổi hành vi tối ưu khi policy
đã bám dưới ngưỡng 0.05. Bài học: reward đơn giản `−dist` là đủ cho task này —
đúng tinh thần "bắt đầu đơn giản, chỉ thêm shaping khi có bằng chứng cần".

## 5. Kết luận

- Custom Gymnasium env + SAC/SB3 giải quyết tốt trajectory tracking 2D:
  success rate ~90% trên cả 3 loại quỹ đạo với cùng hyperparameters.
- SAC vượt PPO về sample efficiency; PPO vượt về wall-clock — lựa chọn tùy
  ràng buộc (số tương tác env đắt hay rẻ).
- Error vector trong observation là yếu tố then chốt cho phép cùng một policy
  bám được target di động với quỹ đạo bất kỳ.

## 6. Reproduce

```bash
pip install -r requirements.txt
pytest tests/ -q                                  # 22 tests
sbatch --job-name=sac_circle scripts/submit_train.sh configs/sac_circle.yaml
python eval.py --run-dir outputs/runs/<run>       # metrics + video overlay
python scripts/plot_comparison.py --runs <run1> <run2> --labels a b --out out.png
```
