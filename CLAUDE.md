# Robot Arm 2D — Trajectory Tracking với Deep RL

## 1. Mục tiêu đề tài
Huấn luyện một **2-link planar robot arm** học policy để **end-effector bám theo quỹ đạo target** (điểm cố định → đường tròn → hình số 8 → spline tự vẽ) bằng Deep Reinforcement Learning (continuous control).

Đây là đồ án môn **Deep Reinforcement Learning**. Deliverable gồm: custom environment, agent đã train (SAC + PPO để so sánh), video demo, metrics, và báo cáo.

## 2. Stack & môi trường
- Python 3.11, venv của project: `.venv/` (activate bằng `source .venv/bin/activate`)
- `gymnasium` (KHÔNG dùng `gym` cũ)
- `stable-baselines3[extra]` (KHÔNG tự implement thuật toán từ đầu)
- `torch` với CUDA (train trên GPU node của cluster)
- `matplotlib` (render + video), `pyyaml` (config), `tensorboard` (logging)
- `pytest` (test env) — chạy với `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (plugin `dash` hỏng trong user site-packages làm crash collection)

Pin version trong `requirements.txt`. **Luôn test `import` sạch trước khi train** vì SB3 ↔ Gymnasium hay mismatch API.

### Train trên cluster SLURM (BẮT BUỘC)
Đây là login node của HPC cluster — **KHÔNG có GPU và KHÔNG train trực tiếp trên login node**. Mọi lần train đều submit job SLURM, mọi sbatch script nằm trong `scripts/`:
```bash
sbatch scripts/submit_train.sh   # submit từ project root; partition main-cpu (MLP nhỏ: CPU nhanh hơn GPU)
squeue -u $USER                  # xem trạng thái job
tail -f outputs/slurm/sac_reach_<jobid>.log
```
- `scripts/submit_train.sh` activate `.venv`, in thông tin GPU, chạy `pytest` (fail thì abort, không train), rồi chạy `train.py --run-dir outputs/runs/<job_name>_<job_id>`.
- **Mỗi session train có folder riêng** `outputs/runs/<run>/` chứa: `config.yaml` (copy config đã dùng), `checkpoints/`, `tb/` (tensorboard), và log SLURM được copy vào cuối job. Không ghi đè output giữa các run.
- Log SLURM gốc: `outputs/slurm/`. Test nhanh / random rollout / render vài giây thì được chạy trên login node; train thì không.

## 3. Cấu trúc project
```
robot-arm-rl/
├── CLAUDE.md
├── requirements.txt
├── robot_arm/
│   ├── __init__.py
│   ├── kinematics.py      # Forward kinematics 2-link
│   ├── env.py             # RobotArm2DEnv (Gymnasium API)
│   ├── trajectories.py    # circle / figure-8 / spline generators
│   └── render.py          # matplotlib render + GIF/MP4 export
├── configs/
│   ├── sac.yaml
│   └── ppo.yaml
├── train.py               # đọc config, train, log tensorboard, save checkpoint
├── scripts/
│   └── submit_train.sh    # sbatch script — cách duy nhất để train (SLURM)
├── eval.py                # metrics + video overlay
├── tests/
│   └── test_env.py        # kiểm tra API, shape, reward, random rollout
└── outputs/               # gitignored
    ├── runs/<run>/        # 1 folder / session train: config.yaml, checkpoints/, tb/
    └── slurm/             # log stdout/stderr của job SLURM
```

## 4. Đặc tả Environment (bám chuẩn Gymnasium)

`RobotArm2DEnv(gymnasium.Env)` phải implement đúng: `reset(seed, options) -> (obs, info)` và `step(action) -> (obs, reward, terminated, truncated, info)` — **5-tuple, không phải 4-tuple**.

**State / observation** (`Box`, dtype float32):
```
[cos θ1, sin θ1, cos θ2, sin θ2,   # góc khớp (dùng cos/sin tránh wrap-around 2π)
 θ1_dot, θ2_dot,                    # vận tốc khớp
 ex, ey,                            # vị trí target hiện tại
 ex - px, ey - py]                  # error vector (target − end-effector)
```

**Action** (`Box`, low=-1, high=1, shape=(2,)): điều khiển 2 khớp. Mặc định **kinematic control** (Δangle = action × max_step), có flag `dynamics_mode` để bật torque control (Euler-Lagrange) ở giai đoạn nâng cao.

**Reward** — bắt đầu ĐƠN GIẢN, thêm penalty dần. Không reward-shaping phức tạp ngay từ đầu:
```python
dist = ||p_ee - target||
reward  = -dist                      # term chính, làm việc trước
reward += -0.01 * ||action||         # phạt effort (smoothness) — thêm ở slice sau
reward += -0.001 * ||θ_dot||         # phạt giật — thêm ở slice sau
if dist < 0.05: reward += 1.0        # bonus bám tốt — thêm ở slice sau
```
Mọi hệ số reward nằm trong config, **không hardcode**.

**Episode**: `truncated` khi hết `max_steps` (vd 200). `terminated` mặc định False (tracking là task liên tục).

## 5. Conventions (BẮT BUỘC tuân thủ)
- Gymnasium API thuần, 5-tuple step. Đăng ký env qua `gymnasium.register` nếu cần.
- State dùng `[cos θ, sin θ]`, không truyền `θ` thô.
- **Mọi hyperparameter → `configs/*.yaml`**, code chỉ đọc config. Không magic number.
- Type hint đầy đủ, docstring ngắn gọn cho mỗi hàm/class.
- Reward terms tách riêng, log từng term vào `info` để debug được.
- Seed mọi thứ (env, numpy, torch) để reproduce.
- Không viết code >150 dòng/file một lần; chia nhỏ.

## 6. Quy trình làm việc theo SLICE
Làm **từng slice một**, không làm hết cùng lúc. RL rất khó debug — phải kiểm soát từng bước. **Sau mỗi slice chạy được thì git commit.**

- **Slice 1**: `kinematics.py` + `env.py` (chỉ reaching 1 điểm cố định, reward = `-dist`) + `tests/test_env.py` (assert shape, API, random rollout). **CHƯA train.**
- **Slice 2**: `render.py` — vẽ 2 link + end-effector + target, xuất GIF. Chạy random policy 1 episode để mắt kiểm tra env đúng trước khi train.
- **Slice 3**: `train.py` với SAC (SB3), đọc `sac.yaml`, log tensorboard, save checkpoint. Train reaching điểm cố định.
- **Slice 4**: `trajectories.py` — target chạy theo đường tròn; cập nhật state (error vector). Train lại, xem learning curve.
- **Slice 5**: `eval.py` — mean/max tracking error, success rate, video overlay target vs quỹ đạo ee thực. Thêm PPO, so sánh với SAC.
- **Nâng cao (tùy chọn)**: curriculum (reaching → slow → fast trajectory), domain randomization (L1/L2, tốc độ target), obstacle avoidance, baseline IK+PID để so sánh với RL.

## 7. Kỷ luật khi train / debug
- **LUÔN chạy `pytest` + random rollout TRƯỚC khi train.** Đừng train mù rồi 30 phút sau mới biết env sai.
- **Train = submit job SLURM** (`sbatch submit_train.sh`), không chạy `train.py` trực tiếp trên login node (xem mục 2).
- Khi policy không học (reward phẳng): đừng đoán mò. Log `dist`, `||action||`, `θ_dot` từng bước và phân tích env/reward trước, không vội đổi thuật toán.
- Clip action, normalize để tránh reward NaN khi giá trị khớp/torque quá lớn.
- SAC/TD3 sample-efficient → ưu tiên cho continuous control. PPO thêm vào để so sánh cho báo cáo.

## 8. Metrics cho báo cáo
- Learning curve (episode reward theo timesteps), SAC vs PPO.
- Tracking error: mean & max Euclidean distance ee↔target.
- Success rate: % bước có `dist < ε`.
- Smoothness: tổng `||action||` hoặc jerk.
- Video overlay: quỹ đạo target vs quỹ đạo ee thực tế.
- Ablation: bật/tắt từng reward term để chứng minh tác dụng.

## 9. KHÔNG làm
- Không dùng `gym` cũ (chỉ `gymnasium`).
- Không tự implement SAC/PPO từ đầu (dùng SB3).
- Không hardcode hyperparameter trong code.
- Không reward shaping phức tạp ngay slice đầu.
- Không commit `outputs/` (checkpoint, video, log) — thêm vào `.gitignore`.
- Không train khi test env chưa pass.