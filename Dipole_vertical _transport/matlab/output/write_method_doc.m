function write_method_doc(path, argo_mat, history_argo_mat, meta_dir, boa_pden_root, output_root, bbox, crossing_lats, target_lat, intersect_radius_r, match_mode, boa_background_mode, time_window_days, core_min_m, core_max_m, density_variable, z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m)
    fid = fopen(path, 'w');
    fprintf(fid, '# META4.0 + Core Argo 垂直速度重建方法与假定\n\n');
    fprintf(fid, '- Argo 主数据源：`%s`\n', argo_mat);
    fprintf(fid, '- 历史 Argo 速度校准源：`%s`\n', history_argo_mat);
    fprintf(fid, '- META4.0 涡旋源：`%s`\n', meta_dir);
    fprintf(fid, '- BOA 背景位密源：`%s`\n', boa_pden_root);
    fprintf(fid, '- 输出根目录：`%s`\n', output_root);
    fprintf(fid, '- 运行范围：`%.1fE-%.1fE, %.1f-%.1f latitude`\n', bbox(1), bbox(2), bbox(3), bbox(4));
    fprintf(fid, '- 样本选择：正式主流程固定为 crossing。涡旋本体 `%.3gR` 跨过纬线 `%s`，逐纬线输出 `cross_XX_%.3gR` 目录。Argo profile 不再按纬度带预筛，只由 bbox、parking depth、时间窗和 `0-4R` 空间匹配决定。\n', intersect_radius_r, crossing_list_text(crossing_lats), intersect_radius_r);
    fprintf(fid, '\n- Core Argo 限定：`%.0f-%.0f m` parking depth。\n', core_min_m, core_max_m);
    fprintf(fid, '- 时间匹配：Argo profile 与 META 轨迹点相差不超过 `%.1f day`。\n', time_window_days);
    fprintf(fid, '- 匹配模式：`%s`。`nearest` 表示每条 Argo 只归属最近的 `r/R` 涡旋；`all` 表示一条 Argo 可在所有满足时间窗和 `0-4R` 的涡旋坐标系中重复使用。\n', match_mode);
    fprintf(fid, '- 速度来源：固定使用历史文件 `I_Upk/I_Vpk/I_Wpk` 与 TEOS profile 的 `I_PF/I_Time/I_Lon/I_Lat` 近似键匹配；`I_Wpk` 只作为校验字段，不参与 `rebuild_W`。\n');
    fprintf(fid, '- 空间分区：保存 `0-1R`、`1-2R`、`2-4R`，图像网格为 `x/R, y/R = -4..4`。\n');
    fprintf(fid, '- 密度变量：`%s`。正式 `rho0` 来自每条 TEOS Argo profile 在实际 parking depth 处的密度；背景固定为 BOA 多年同月局地密度剖面。\n', density_variable);
    fprintf(fid, '- `z` 几何：正式生产结果固定为 `BOA monthly climatology -> z''_rho -> Cressman composite -> gradient/W`。整体等密面集合、reversal factor 和 zgeometry 只保留在 diagnostics 入口。\n');
    fprintf(fid, '- BOA 背景模式：`%s`。默认按月份平均 `PDen1000_YYYYMM.mat`，对每个 profile 的 `lon/lat/month` 双线性插值得到背景密度剖面。\n', boa_background_mode);
    fprintf(fid, '- `z_rho` 反插值：profile 和 BOA 背景均只使用显式 bracket crossing，不再用全剖面 fallback；有效窗口 `%.0f-%.0f m`，`abs(local_drho_dz) >= %.3g`，bracket 厚度 `<= %.0f m`。\n', z_rho_min_m, z_rho_max_m, min_drho_dz, max_rho_bracket_dz_m);
    fprintf(fid, '- 默认网格化：Cressman objective mapping。权重 `w=(R_c^2-r^2)/(R_c^2+r^2)`，仅使用 `R_c` 内样本；默认 `R_c=0.5R`、每格至少 `3` 个样本。`sample_count` 是原始 bin 覆盖，`mapped_support` 是 Cressman 支撑样本数。\n');
    fprintf(fid, '- 默认输出格式：大匹配表写为 `.mat`；网格写为 `.mat` 和 `.nc`；SUMMARY 写为 `.mat`。CSV 与网格 JSON 默认关闭，可用 `--write-matched-csv`、`--write-summary-csv` 和 `--write-grid-json` 显式打开。\n');
    fprintf(fid, '- 默认绘图：二维 W 图使用 `contourf(..., ''LineStyle'', ''none'')`，只显示填色块，不叠加等值线描边。\n');
    fprintf(fid, '- 深度变量约定：正式代码内部统一使用 `D_rho`，即正深度向下；历史字段名 `z_rho_m/z_rho_bg_m/z_rho_anom_m` 仅为兼容名，物理含义按 `D_rho/D_rho_bg/D''_rho` 解释。\n');
    fprintf(fid, '- W 符号约定：`rebuild_w_m_s`、`term1_m_s`、`term2_m_s` 统一为向上为正，即 `W_up = -D_t`；同时保留 `rebuild_w_raw_depth_positive_m_s` 作为深度向下正公式对照。\n');
    fprintf(fid, '- `c_x_raw` 来自 META track 相邻点中央差分；`u_bg` 为同 crossing 组、同极性、匹配 Core Argo 的 parking drift 纬向均值；`c_x_rel = mean(c_x_raw) - mean(u_bg)`。主图采用 `term1 = +c_x_rel dD''_rho/dx`，`term2 = -[(u_pk-u_bg, v_pk) · grad(D''_rho)]`，`rebuild_W = term1 + term2`，避免传播速度在 term1 和 term2 中重复计入。\n');
    fprintf(fid, '- BOA_Argo 只作为 gridded 密度背景，不直接推导背景速度。\n\n');
    fprintf(fid, '参考：NOAA AOML Argo overview, NOAA Argo best practices, Lin et al. 2019 Remote Sensing, Zhou et al. 2023 JGR Oceans, JAMSTEC Argo gridded products。\n');
    fclose(fid);
end
