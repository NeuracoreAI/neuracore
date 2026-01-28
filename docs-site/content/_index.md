---
title: Neuracore Documentation
layout: hextra-home
---

{{< hextra/hero-badge >}}
  <div class="hx-w-2 hx-h-2 hx-rounded-full hx-bg-primary-400"></div>
  <span>Free, open source</span>
  {{< icon name="arrow-circle-right" attributes="height=14" >}}
{{< /hextra/hero-badge >}}

<div class="hx-mt-6 hx-mb-6">
{{< hextra/hero-headline >}}
  Build Robot Learning&nbsp;<br class="sm:hx-block hx-hidden" />Workflows with Neuracore
{{< /hextra/hero-headline >}}
</div>

<div class="hx-mb-12">
{{< hextra/hero-subtitle >}}
  A powerful robot learning library that enables data collection,&nbsp;<br class="sm:hx-block hx-hidden" />model training, deployment, and real-time inference.
{{< /hextra/hero-subtitle >}}
</div>

<div class="hx-mb-6">
{{< hextra/hero-button text="Get Started" link="docs/getting-started" >}}
{{< hextra/hero-button text="Examples" link="examples" style="outline" >}}
</div>

<div class="hx-mt-6"></div>

{{< hextra/feature-grid >}}
  {{< hextra/feature-card
    title="Streaming Data Logging"
    subtitle="Log robot data with custom data types in real-time. Support for joint positions, camera feeds, language instructions, and more."
    class="hx-aspect-auto md:hx-aspect-[1.1/1] max-md:hx-min-h-[340px]"
    style="background: radial-gradient(ellipse at 50% 80%,rgba(142,53,74,0.15),hsla(0,0%,100%,0));"
  >}}
  {{< hextra/feature-card
    title="Dataset Visualization"
    subtitle="Visualize and synchronize your robot datasets. Browse episodes, inspect data streams, and analyze recordings in the web dashboard."
    class="hx-aspect-auto md:hx-aspect-[1.1/1] max-lg:hx-min-h-[340px]"
    style="background: radial-gradient(ellipse at 50% 80%,rgba(53,142,74,0.15),hsla(0,0%,100%,0));"
  >}}
  {{< hextra/feature-card
    title="Cloud Training"
    subtitle="Train robot learning algorithms on the cloud with distributed multi-GPU support. Built-in algorithms include Diffusion Policy, ACT, Pi0, and more."
    class="hx-aspect-auto md:hx-aspect-[1.1/1] max-md:hx-min-h-[340px]"
    style="background: radial-gradient(ellipse at 50% 80%,rgba(53,74,142,0.15),hsla(0,0%,100%,0));"
  >}}
  {{< hextra/feature-card
    title="Policy Inference"
    subtitle="Deploy trained models locally or remotely. Run inference with automatic data synchronization and action prediction."
  >}}
  {{< hextra/feature-card
    title="Multi-modal Support"
    subtitle="Work with images, joint states, language instructions, point clouds, poses, and custom 1D data. Define your own data types."
  >}}
  {{< hextra/feature-card
    title="Dataset Importer"
    subtitle="Import datasets from RLDS, LeRobot, and TFDS formats. Automatic format detection and flexible data mapping."
  >}}
{{< /hextra/feature-grid >}}
