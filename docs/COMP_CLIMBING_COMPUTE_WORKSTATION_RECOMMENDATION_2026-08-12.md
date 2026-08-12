# Comp Climbing compute workstation recommendation

Decision date: 2026-08-12. Prices are approximate Canadian-market planning
ranges, not a purchase authorization or quote.

## Decision

Keep the HP Spectre x360 as the mobile computer and add an always-on desktop
worker. Do not replace the present single-computer bottleneck with another
expensive laptop that must travel and cannot run unattended.

Observed Spectre hardware:

- Intel Core i7-10510U, 4 cores / 8 threads;
- 16 GB DDR4-2667, with 2.17 GB available during inspection;
- NVIDIA MX250, 2 GB VRAM;
- 512 GB Intel Optane/SSD;
- 37.8 GB configured page file.

The project is currently constrained more by CPU, physical RAM, storage and
single-machine availability than by VRAM. The large page file and low free RAM
make paging a credible source of severe slowdown during concurrent analytics.

## Recommended desktop

- 16-core Ryzen 9 9950X-class CPU or current equivalent;
- 128 GB RAM preferred, 96 GB acceptable;
- 4 TB NVMe minimum, with separate backup/data storage;
- NVIDIA CUDA GPU:
  - 16 GB is the current-project baseline;
  - 24–32 GB is the preferred durable configuration;
  - 48–96 GB is not cost-effective without a specific approved local large-
    model or large-video-model workload;
- high-quality cooling and power supply, wired networking and UPS;
- automatic restart after power loss and private authenticated remote access.

Approximate planning tiers:

1. CPU/data-science value: 16-core CPU, 128 GB RAM, 4 TB NVMe, RTX 5070 Ti
   16 GB, approximately CAD 3,500–4,500.
2. Durable computer-vision/local-AI: same foundation with RTX 5090 32 GB,
   approximately CAD 5,000–7,000.
3. Used-value alternative: a carefully tested RTX 3090 24 GB can provide high
   VRAM per dollar, but power use, warranty and always-on reliability are worse.

Do not trade away CPU, system RAM or storage merely to reach a 32 GB GPU.

## Expected value by workload

- Pandas/Parquet, identity rebuilds, repository work and Streamlit: VRAM has
  essentially no direct effect. CPU, RAM and SSD dominate.
- Current NumPy/CPU simulations: a modern cooled 16-core desktop should often
  be roughly 4–10 times faster than this four-core mobile CPU, with larger gains
  when paging or thermal throttling is eliminated. Measure exact pipelines
  after acquisition; this is a planning range, not a benchmark claim.
- Rating/graph models: most proposed models fit comfortably in 16 GB if
  implemented for GPU execution. Their statistical validity is a larger
  constraint than GPU capacity.
- Video OCR/detection/tagging: a modern NVIDIA desktop GPU can enable batched
  decoding/inference that is plausibly an order of magnitude faster than the
  MX250. This remains a secondary lane until its automation beats human effort.
- Local language/vision models: VRAM primarily determines which model and
  context fit without CPU offload. Moving from 2 GB to 16 GB is transformative;
  24–32 GB is materially more capable. Extra unused VRAM provides no inherent
  speedup, and frontier Gemini review remains more efficient in Vertex.

## Operating model

The desktop runs local rebuilds, tests, simulations, computer vision and
development services. The Spectre is the mobile terminal. Cloud Run/Vertex
handles bounded unattended or frontier-model work where it is cheaper or more
reliable. Every critical pipeline remains reproducible from versioned inputs;
the desktop must not become an irreplaceable data silo.

Before purchase, compare exact Canadian configurations for expandability,
cooling, noise, warranty, RAM slot use, GPU power limits and remote-service
reliability. A consumer gaming tower can be better value than an OEM
workstation, but only if it uses standard replaceable parts and adequate power
and cooling.
