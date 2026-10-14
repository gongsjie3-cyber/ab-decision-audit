# NOTICE

## 代码与文档

本仓库的源码与分析文档采用 MIT 许可证，全文见 [LICENSE](./LICENSE)。

## 数据

本项目使用的数据来自 **Upworthy Research Archive**（<https://osf.io/jd64p/>），
以 **CC BY 4.0** 授权发布，**不属于**上述 MIT 许可范围。使用时必须署名：

> Matias, J., Munger, K., Aubin Le Quere, M., & Ebersole, C. (2021).
> *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media.*
> Scientific Data, 8(1), 155. <https://doi.org/10.1038/s41597-021-00934-7>

原始数据不随本仓库分发，请运行 `python scripts/fetch_data.py` 自行下载（含 sha256 校验）。

## 致谢

感谢 Upworthy 与 Cornell University 将该档案公开，也感谢档案团队 2024 年主动披露
随机化失衡问题——本项目的模块 5 正是在这一披露的基础上，进一步量化了该问题对决策的
实际影响。
