/**
 * Plotly trace/layout builders for visualizer /plots payloads (plotKind + raw data).
 * Styling and presentation live here; the API sends series and axis metadata only.
 */
(function (global) {
    'use strict';

    const STORAGE_TANK_KEYS = [
        'tank1-depth1', 'tank1-depth2', 'tank1-depth3', 'tank1-depth4',
        'tank2-depth1', 'tank2-depth2', 'tank2-depth3', 'tank2-depth4',
        'tank3-depth1', 'tank3-depth2', 'tank3-depth3', 'tank3-depth4'
    ];

    const BUFFER_DEPTH_KEYS = ['buffer-depth1', 'buffer-depth2', 'buffer-depth3', 'buffer-depth4'];

    /**
     * Matplotlib `plt.get_cmap('coolwarm', N)` discrete colors (Moreland coolwarm, mpl 3.8 segment data).
     * Index 0 = cool (blue), N-1 = warm (red). Matches server-side buffer/storage traces.
     */
    const BUFFER_COOLWARM_4 = ['#3b4cc0', '#aac7fd', '#f7b89c', '#b40426'];

    const STORAGE_COOLWARM_12 = [
        '#3b4cc0', '#5774e0', '#7598f6', '#95b7ff', '#b4cdfa', '#d1dae9',
        '#e8d6cc', '#f5c1a8', '#f6a384', '#ea7c61', '#d44e41', '#b40426'
    ];

    function bufferDepthColor(key) {
        const idx = BUFFER_DEPTH_KEYS.indexOf(key);
        if (idx < 0) return '#808080';
        return BUFFER_COOLWARM_4[3 - idx];
    }

    function storageTankColor(key) {
        const i = STORAGE_TANK_KEYS.indexOf(key);
        if (i < 0) return '#808080';
        return STORAGE_COOLWARM_12[11 - i];
    }

    function milliCtoF(v) {
        return (v / 1000) * 9 / 5 + 32;
    }

    function theme(isDarkMode) {
        return {
            fontColor: isDarkMode ? '#b5b5b5' : 'rgb(42,63,96)',
            bg: isDarkMode ? '#1b1b1c' : 'white',
            gridColor: isDarkMode ? '#424242' : 'LightGray',
            lineMuted: isDarkMode ? '#f0f0f0' : '#5e5e5e'
        };
    }

    function latePersistenceShapes(periodsMs, toNyLocalIso) {
        return (periodsMs || []).map(([x0ms, x1ms]) => ({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: toNyLocalIso(x0ms),
            x1: toNyLocalIso(x1ms),
            y0: 0,
            y1: 1,
            fillcolor: 'red',
            opacity: 0.15,
            layer: 'below',
            line: { width: 0 }
        }));
    }

    function xRangeFromPayload(payload, toNyLocalIso) {
        if (!payload.x_range_ms || payload.x_range_ms.length !== 2) return undefined;
        return [toNyLocalIso(payload.x_range_ms[0]), toNyLocalIso(payload.x_range_ms[1])];
    }

    function buildHeatpump(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const channels = payload.channels || {};
        const showPoints = selectedChannels.includes('show-points');
        const mode = showPoints ? 'lines+markers' : 'lines';
        const take = (channelKey) => {
            if (!selectedChannels.includes(channelKey)) return null;
            const c = channels[channelKey];
            if (!c || !Array.isArray(c.times) || c.times.length === 0) return null;
            return c;
        };
        let plottingTemperatures = false;
        let plottingPower = false;
        const traces = [];
        let c = take('hp-lwt');
        if (c) {
            plottingTemperatures = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map(milliCtoF),
                mode,
                opacity: 0.7,
                line: { color: '#d62728', dash: 'solid' },
                name: 'HP LWT',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        c = take('hp-ewt');
        if (c) {
            plottingTemperatures = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map(milliCtoF),
                mode,
                opacity: 0.7,
                line: { color: '#1f77b4', dash: 'solid' },
                name: 'HP EWT',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        const yAxisPower = plottingTemperatures ? 'y2' : 'y';
        c = take('hp-odu-pwr');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 1000),
                mode,
                opacity: 0.7,
                line: { color: '#2ca02c', dash: 'solid', shape: 'hv' },
                name: 'HP outdoor power',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>'
            });
        }
        c = take('hp-idu-pwr');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 1000),
                mode,
                opacity: 0.7,
                line: { color: '#ff7f0e', dash: 'solid', shape: 'hv' },
                name: 'HP indoor power',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>'
            });
        }
        c = take('oil-boiler-pwr');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 100),
                mode,
                opacity: 0.7,
                line: { color: isDarkMode ? '#f0f0f0' : '#5e5e5e', dash: 'solid', shape: 'hv' },
                name: 'Oil boiler power x10',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}/10 kW<extra></extra>'
            });
        }
        c = take('primary-flow');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 100),
                mode,
                opacity: 0.4,
                line: { color: 'purple', dash: 'solid', shape: 'hv' },
                name: 'Primary pump flow',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} GPM<extra></extra>'
            });
        }
        c = take('primary-pump-pwr');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => (v / 1000) * 100),
                mode,
                opacity: 0.7,
                line: { color: 'pink', dash: 'solid', shape: 'hv' },
                name: 'Primary pump power x100',
                yaxis: yAxisPower,
                visible: 'legendonly',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}/100 kW<extra></extra>'
            });
        }
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const layout = {
            title: { text: 'Heat pump', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            yaxis2: {
                mirror: true,
                ticks: 'outside',
                zeroline: false,
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                overlaying: 'y',
                side: 'right'
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        if (plottingPower && plottingTemperatures) {
            layout.yaxis.title = 'Temperature [F]';
            layout.yaxis.range = [0, 260];
            layout.yaxis2.title = 'Power [kW] or Flow [GPM]';
            layout.yaxis2.range = [0, 35];
        } else if (plottingTemperatures && !plottingPower) {
            layout.yaxis.title = 'Temperature [F]';
        } else if (plottingPower && !plottingTemperatures) {
            layout.yaxis.title = 'Power [kW] or Flow [GPM]';
            layout.yaxis.range = [0, 10];
        }
        return { traces, layout };
    }

    function buildDistribution(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const channels = payload.channels || {};
        const showPoints = selectedChannels.includes('show-points');
        const mode = showPoints ? 'lines+markers' : 'lines';
        const take = (key) => {
            if (!selectedChannels.includes(key)) return null;
            const c = channels[key];
            if (!c || !Array.isArray(c.times) || c.times.length === 0) return null;
            return c;
        };
        let plottingTemperatures = false;
        let plottingPower = false;
        const traces = [];
        let c = take('dist-swt');
        if (c) {
            plottingTemperatures = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map(milliCtoF),
                mode,
                opacity: 0.7,
                line: { color: '#d62728', dash: 'solid' },
                name: 'Distribution SWT',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        c = take('dist-rwt');
        if (c) {
            plottingTemperatures = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map(milliCtoF),
                mode,
                opacity: 0.7,
                line: { color: '#1f77b4', dash: 'solid' },
                name: 'Distribution RWT',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        const yAxisPower = plottingTemperatures ? 'y2' : 'y';
        c = take('dist-flow');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 100),
                mode,
                opacity: 0.4,
                line: { color: 'purple', dash: 'solid', shape: 'hv' },
                name: 'Distribution flow',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} GPM<extra></extra>'
            });
        }
        c = take('dist-pump-pwr');
        if (c) {
            plottingPower = true;
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y: c.values.map((v) => v / 10),
                mode,
                opacity: 0.7,
                line: { color: 'pink', dash: 'solid', shape: 'hv' },
                name: 'Distribution pump power /10',
                yaxis: yAxisPower,
                visible: 'legendonly',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}*10 W<extra></extra>'
            });
        }
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const layout = {
            title: { text: 'Distribution', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            yaxis2: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                overlaying: 'y',
                side: 'right',
                showgrid: false
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        if (plottingTemperatures && plottingPower) {
            layout.yaxis.title = 'Temperature [F]';
            layout.yaxis.range = [0, 260];
            layout.yaxis2.title = 'Flow [GPM] or Power [W]';
            layout.yaxis2.range = [0, 20];
        } else if (plottingTemperatures && !plottingPower) {
            layout.yaxis.title = 'Temperature [F]';
            layout.yaxis.range = [0, 260];
        } else if (plottingPower && !plottingTemperatures) {
            layout.yaxis.title = 'Flow [GPM] or Power [W]';
            layout.yaxis.range = [0, 20];
        }
        return { traces, layout };
    }

    function buildHeatcalls(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const traces = [];
        const zoneAxisCount = payload.zone_axis_count || 0;
        if (selectedChannels.includes('zone-heat-calls')) {
            const threshold = payload.whitewire_threshold;
            const zoneColors = payload.zone_colors || [];
            const showPoints = selectedChannels.includes('show-points');
            for (const z of payload.zones || []) {
                const ww_times = z.times;
                const ww_values = z.values;
                if (!ww_times || !ww_values || !ww_times.length) continue;
                const zone_number = z.zone_number;
                const zone_color = zoneColors[zone_number - 1] || '#888888';
                const active = ww_values.map((v) => Math.abs(v) > threshold);
                const anyActive = active.some(Boolean);
                if (!anyActive) {
                    traces.push({
                        type: 'scatter',
                        x: [null],
                        y: [null],
                        mode: 'lines',
                        line: { color: zone_color, width: 2 },
                        name: z.legend_name
                    });
                    continue;
                }
                const starts = [];
                const ends = [];
                for (let i = 0; i < active.length; i++) {
                    const prevActive = i === 0 ? true : !active[i - 1];
                    if (active[i] && prevActive) starts.push(i);
                    const nextInactive = i === active.length - 1 ? true : !active[i + 1];
                    if (active[i] && nextInactive) ends.push(i);
                }
                const fill_x = [];
                const fill_y = [];
                const edge_x = [];
                const edge_y = [];
                for (let k = 0; k < starts.length; k++) {
                    const s_idx = starts[k];
                    const e_idx = ends[k];
                    const e_time_idx = Math.min(e_idx + 1, ww_times.length - 1);
                    const x0 = ww_times[s_idx];
                    const x1 = ww_times[e_time_idx];
                    fill_x.push(x0, x0, x1, x1, null);
                    fill_y.push(zone_number - 1, zone_number, zone_number, zone_number - 1, null);
                    edge_x.push(x0, x0, null, x1, x1, null);
                    edge_y.push(zone_number - 1, zone_number, null, zone_number - 1, zone_number, null);
                }
                traces.push({
                    type: 'scatter',
                    x: fill_x.map(toNyLocalIso),
                    y: fill_y,
                    mode: 'lines',
                    fill: 'toself',
                    line: { color: zone_color, width: 0 },
                    fillcolor: zone_color,
                    opacity: 0.2,
                    showlegend: false,
                    hoverinfo: 'skip'
                });
                traces.push({
                    type: 'scatter',
                    x: edge_x.map(toNyLocalIso),
                    y: edge_y,
                    mode: 'lines',
                    line: { color: zone_color, width: 2 },
                    opacity: 0.7,
                    showlegend: false,
                    hovertemplate: '%{x|%H:%M:%S}<extra></extra>'
                });
                if (showPoints) {
                    const activeIdx = [];
                    for (let i = 0; i < active.length; i++) if (active[i]) activeIdx.push(i);
                    traces.push({
                        type: 'scatter',
                        x: activeIdx.map((i) => toNyLocalIso(ww_times[i])),
                        y: activeIdx.map(() => zone_number - 0.5),
                        mode: 'markers',
                        marker: { size: 4, color: zone_color, opacity: 0.6 },
                        showlegend: false,
                        hovertemplate: '%{x|%H:%M:%S}<extra></extra>'
                    });
                }
                traces.push({
                    type: 'scatter',
                    x: [null],
                    y: [null],
                    mode: 'lines',
                    line: { color: zone_color, width: 2 },
                    name: z.legend_name
                });
            }
        }
        const layout = {
            title: { text: 'Heat calls', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                range: [-0.5, zoneAxisCount * 1.3],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor,
                tickvals: Array.from({ length: zoneAxisCount + 1 }, (_, i) => i)
            },
            yaxis2: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    function buildZones(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const showPoints = selectedChannels.includes('show-points');
        const mode = showPoints ? 'lines+markers' : 'lines';
        const zoneColors = payload.zone_colors || [];
        let min_zones = 45;
        let max_zones = 80;
        let min_oat = 70;
        let max_oat = 80;
        const traces = [];
        for (const z of payload.zones || []) {
            const col = zoneColors[z.zone_digit - 1] || '#888888';
            if (z.temp) {
                const c = z.temp;
                traces.push({
                    type: 'scatter',
                    x: c.times.map(toNyLocalIso),
                    y: c.values.map((v) => v / 1000),
                    mode,
                    opacity: 0.7,
                    line: { color: col, dash: 'solid', shape: 'hv' },
                    name: c.legend_suffix,
                    hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
                });
                min_zones = Math.min(min_zones, ...c.values.map((v) => v / 1000));
                max_zones = Math.max(max_zones, ...c.values.map((v) => v / 1000));
            }
            if (z.set) {
                const c = z.set;
                traces.push({
                    type: 'scatter',
                    x: c.times.map(toNyLocalIso),
                    y: c.values.map((v) => v / 1000),
                    mode,
                    opacity: 0.7,
                    line: { color: col, dash: 'dash', shape: 'hv' },
                    name: c.legend_suffix,
                    showlegend: false,
                    hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
                });
                min_zones = Math.min(min_zones, ...c.values.map((v) => v / 1000));
                max_zones = Math.max(max_zones, ...c.values.map((v) => v / 1000));
            }
        }
        let hasOat = false;
        if (selectedChannels.includes('oat') && payload.oat && payload.oat.times && payload.oat.times.length) {
            hasOat = true;
            const o = payload.oat;
            traces.push({
                type: 'scatter',
                x: o.times.map(toNyLocalIso),
                y: o.values.map(milliCtoF),
                mode,
                opacity: 0.8,
                line: { color: isDarkMode ? 'gray' : '#d6d6d6', dash: 'solid', shape: 'hv' },
                name: 'Outside air',
                yaxis: 'y2',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
            const fvals = o.values.map(milliCtoF);
            min_oat = Math.min(...fvals);
            max_oat = Math.max(...fvals);
        }
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const layout = {
            title: { text: 'Zones', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                title: 'Zone temperature [F]',
                range: [min_zones - 30, max_zones + 20],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            yaxis2: {
                title: hasOat ? 'Outside air temperature [F]' : undefined,
                range: hasOat ? [min_oat - 2, max_oat + 20] : undefined,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                overlaying: 'y',
                side: 'right',
                zeroline: false,
                showgrid: false
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    function buildBuffer(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const showPoints = selectedChannels.includes('show-points');
        const mode = showPoints ? 'lines+markers' : 'lines';
        const deci = payload.use_decicelsius_depth_scale;
        const depthY = (v) => (deci ? v / 100 : milliCtoF(v));
        let minT = 1e15;
        let maxT = 0;
        const traces = [];
        if (selectedChannels.includes('buffer-depths')) {
            for (const row of payload.buffer_depths || []) {
                const col = bufferDepthColor(row.key);
                const y = row.values.map(depthY);
                minT = Math.min(minT, ...y);
                maxT = Math.max(maxT, ...y);
                traces.push({
                    type: 'scatter',
                    x: row.times.map(toNyLocalIso),
                    y,
                    mode,
                    opacity: 0.7,
                    name: row.key.replace('buffer-', ''),
                    line: { color: col, dash: 'solid' },
                    hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
                });
            }
        }
        const pipeY = (v) => milliCtoF(v);
        if (selectedChannels.includes('buffer-hot-pipe') && payload.buffer_hot_pipe) {
            const h = payload.buffer_hot_pipe;
            const y = h.values.map(pipeY);
            minT = Math.min(minT, ...y);
            maxT = Math.max(maxT, ...y);
            traces.push({
                type: 'scatter',
                x: h.times.map(toNyLocalIso),
                y,
                mode,
                opacity: 0.7,
                name: 'Hot pipe',
                line: { color: '#d62728', dash: 'solid', shape: 'hv' },
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        if (selectedChannels.includes('buffer-cold-pipe') && payload.buffer_cold_pipe) {
            const c = payload.buffer_cold_pipe;
            const y = c.values.map(pipeY);
            minT = Math.min(minT, ...y);
            maxT = Math.max(maxT, ...y);
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y,
                mode,
                opacity: 0.7,
                name: 'Cold pipe',
                line: { color: '#1f77b4', dash: 'solid', shape: 'hv' },
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        if (minT > 1e14) {
            minT = 0;
            maxT = 100;
        }
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const layout = {
            title: { text: 'Buffer', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                title: 'Temperature [F]',
                range: [minT - 15, maxT + 30],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    function buildStorage(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const showPoints = selectedChannels.includes('show-points');
        const mode = showPoints ? 'lines+markers' : 'lines';
        const deci = payload.use_decicelsius_depth_scale;
        const depthY = (v) => (deci ? v / 100 : milliCtoF(v));
        let min_store_temp = 1e15;
        let max_store_temp = 0;
        let plotting_temperatures = false;
        let plotting_power = false;
        const traces = [];

        if (selectedChannels.includes('storage-depths')) {
            plotting_temperatures = true;
            for (const row of payload.tank_depths || []) {
                const col = storageTankColor(row.key);
                const y = row.values.map(depthY);
                min_store_temp = Math.min(min_store_temp, ...y);
                max_store_temp = Math.max(max_store_temp, ...y);
                traces.push({
                    type: 'scatter',
                    x: row.times.map(toNyLocalIso),
                    y,
                    mode,
                    opacity: 0.7,
                    name: row.key.replace('storage-', ''),
                    line: { color: col, dash: 'solid' },
                    hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
                });
            }
        }
        const pipeY = (v) => milliCtoF(v);
        if (selectedChannels.includes('store-hot-pipe') && payload.store_hot_pipe) {
            plotting_temperatures = true;
            const h = payload.store_hot_pipe;
            const y = h.values.map(pipeY);
            min_store_temp = Math.min(min_store_temp, ...y);
            max_store_temp = Math.max(max_store_temp, ...y);
            traces.push({
                type: 'scatter',
                x: h.times.map(toNyLocalIso),
                y,
                mode,
                opacity: 0.7,
                name: 'Hot pipe',
                line: { color: '#d62728', dash: 'solid', shape: 'hv' },
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }
        if (selectedChannels.includes('store-cold-pipe') && payload.store_cold_pipe) {
            plotting_temperatures = true;
            const c = payload.store_cold_pipe;
            const y = c.values.map(pipeY);
            min_store_temp = Math.min(min_store_temp, ...y);
            max_store_temp = Math.max(max_store_temp, ...y);
            traces.push({
                type: 'scatter',
                x: c.times.map(toNyLocalIso),
                y,
                mode,
                opacity: 0.7,
                name: 'Cold pipe',
                line: { color: '#1f77b4', dash: 'solid' },
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>'
            });
        }

        const yAxisPower = plotting_temperatures ? 'y2' : 'y';
        if (selectedChannels.includes('store-pump-pwr') && payload.store_pump_pwr) {
            plotting_power = true;
            const s = payload.store_pump_pwr;
            traces.push({
                type: 'scatter',
                x: s.times.map(toNyLocalIso),
                y: s.values,
                mode,
                opacity: 0.7,
                line: { color: 'pink', dash: 'solid', shape: 'hv' },
                name: 'Storage pump power x1000',
                yaxis: yAxisPower,
                visible: 'legendonly',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}/1000 kW<extra></extra>'
            });
        }
        if (selectedChannels.includes('store-flow') && payload.store_flow) {
            plotting_power = true;
            const s = payload.store_flow;
            traces.push({
                type: 'scatter',
                x: s.times.map(toNyLocalIso),
                y: s.values.map((v) => (v / 100) * 10),
                mode,
                opacity: 0.4,
                line: { color: 'purple', dash: 'solid', shape: 'hv' },
                name: 'Storage pump flow x10',
                yaxis: yAxisPower,
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f}/10 GPM<extra></extra>'
            });
        }
        if (selectedChannels.includes('store-energy') && payload.usable_energy) {
            plotting_power = true;
            const u = payload.usable_energy;
            traces.push({
                type: 'scatter',
                x: u.times.map(toNyLocalIso),
                y: u.values.map((v) => v / 1000),
                mode,
                opacity: 0.4,
                line: { color: '#2ca02c', dash: 'solid' },
                name: 'Usable',
                yaxis: yAxisPower,
                visible: 'legendonly',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>'
            });
            if (payload.required_energy) {
                const r = payload.required_energy;
                traces.push({
                    type: 'scatter',
                    x: r.times.map(toNyLocalIso),
                    y: r.values.map((v) => v / 1000),
                    mode,
                    opacity: 0.4,
                    line: { color: '#2ca02c', dash: 'dash' },
                    name: 'Required',
                    yaxis: yAxisPower,
                    visible: 'legendonly',
                    hovertemplate: '%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>'
                });
            }
        }

        let max_required_energy = 0;
        if (
            selectedChannels.includes('required-energy') &&
            selectedChannels.includes('usable-energy') &&
            payload.required_energy
        ) {
            max_required_energy = Math.max(...payload.required_energy.values.map((v) => v / 1000)) * 4;
        }
        let max_usable_energy = 0;
        if (selectedChannels.includes('usable-energy') && payload.usable_energy) {
            max_usable_energy = Math.max(...payload.usable_energy.values.map((v) => v / 1000));
        }
        let max_store_pump_pwr = 0;
        if (selectedChannels.includes('store-pump-pwr') && payload.store_pump_pwr) {
            max_store_pump_pwr = Math.max(...payload.store_pump_pwr.values);
        }
        let max_store_flow = 0;
        if (selectedChannels.includes('store-flow') && payload.store_flow) {
            max_store_flow = Math.max(...payload.store_flow.values.map((v) => (v / 100) * 10));
        }
        const max_power = Math.max(max_required_energy, max_usable_energy, max_store_pump_pwr, max_store_flow, 15);

        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const layout = {
            title: { text: 'Storage', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            yaxis2: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                overlaying: 'y',
                side: 'right',
                zeroline: false,
                showgrid: false
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };

        if (plotting_temperatures && min_store_temp > 1e14) {
            min_store_temp = 20;
            max_store_temp = 80;
        }
        if (plotting_temperatures && plotting_power) {
            layout.yaxis.title = 'Temperature [F]';
            layout.yaxis.range = [min_store_temp - 80, max_store_temp + 80];
            layout.yaxis2.title = 'GPM, kW, or kWh';
            layout.yaxis2.range = [-1, max_power * 3.5];
        } else if (plotting_temperatures && !plotting_power) {
            let lo = min_store_temp;
            if (lo < 0) lo = 20;
            layout.yaxis.title = 'Temperature [F]';
            layout.yaxis.range = [lo - 20, max_store_temp + 60];
        } else if (plotting_power && !plotting_temperatures) {
            layout.yaxis.title = 'GPM, kW, or kWh';
        }

        return { traces, layout };
    }

    const TOP_STATE_COLORS = {
        LocalControl: '#EF553B',
        LeafTransactiveNode: '#00CC96',
        Admin: '#636EFA'
    };

    function buildTopState(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const ts = payload.top_states || {};
        const traces = [];
        if (Object.keys(ts).length) {
            const all = ts.all;
            if (all && all.times && all.times.length) {
                traces.push({
                    type: 'scatter',
                    x: all.times.map(toNyLocalIso),
                    y: all.values,
                    mode: 'lines',
                    line: { color: t.lineMuted, width: 2, shape: 'hv' },
                    opacity: 0.3,
                    showlegend: false
                });
            }
            for (const state of Object.keys(ts)) {
                if (state === 'all' || !TOP_STATE_COLORS[state]) continue;
                const s = ts[state];
                if (!s || !s.times || !s.times.length) continue;
                traces.push({
                    type: 'scatter',
                    x: s.times.map(toNyLocalIso),
                    y: s.values,
                    mode: 'markers',
                    marker: { color: TOP_STATE_COLORS[state], size: 10 },
                    opacity: 0.8,
                    name: state,
                    hovertemplate: '%{x|%H:%M:%S}<extra></extra>'
                });
            }
        }
        const nStates = Object.keys(ts).length;
        const yHi = nStates > 0 ? nStates - 1 + 0.2 : 0.2;
        const tickTop = Math.max(nStates - 1, 0);
        const layout = {
            title: { text: 'Top State', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                range: [-0.6, yHi],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor,
                tickvals: Array.from({ length: tickTop }, (_, i) => i)
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    const HA_STATE_COLORS = {
        HpOffStoreDischarge: '#EF553B',
        HpOffStoreOff: '#00CC96',
        HpOnStoreOff: '#636EFA',
        HpOnStoreCharge: '#feca52',
        Initializing: '#a3a3a3',
        StratBoss: '#ee93fa',
        Dormant: '#4f4f4f',
        EverythingOff: '#4f4f4f',
        Other: '#ff0000'
    };

    function buildHaState(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const lc = payload.lc_states || {};
        const traces = [];
        if (Object.keys(lc).length) {
            const all = lc.all;
            if (all && all.times && all.times.length) {
                traces.push({
                    type: 'scatter',
                    x: all.times.map(toNyLocalIso),
                    y: all.values,
                    mode: 'lines',
                    line: { color: t.lineMuted, width: 2, shape: 'hv' },
                    opacity: 0.3,
                    showlegend: false
                });
            }
            for (const state of Object.keys(lc)) {
                if (state === 'all') continue;
                const s = lc[state];
                if (!s || !s.times || !s.times.length) continue;
                const col = HA_STATE_COLORS[state] || HA_STATE_COLORS.Other;
                traces.push({
                    type: 'scatter',
                    x: s.times.map(toNyLocalIso),
                    y: s.values,
                    mode: 'markers',
                    marker: { color: col, size: 10 },
                    opacity: 0.8,
                    name: state,
                    hovertemplate: '%{x|%H:%M:%S}<extra></extra>'
                });
            }
        }
        const layout = {
            title: { text: 'LocalControl State', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                range: [-0.6, 8 - 0.8],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor,
                tickvals: [0, 1, 2, 3, 4, 5]
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    const AA_STATE_COLORS = {
        HpOffStoreDischarge: '#EF553B',
        HpOffStoreOff: '#00CC96',
        HpOnStoreOff: '#636EFA',
        HpOnStoreCharge: '#feca52',
        Initializing: '#a3a3a3',
        StratBoss: '#ee93fa',
        Dormant: '#4f4f4f'
    };

    function buildAaState(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const la = payload.la_states || {};
        const traces = [];
        if (Object.keys(la).length) {
            const all = la.all;
            if (all && all.times && all.times.length) {
                traces.push({
                    type: 'scatter',
                    x: all.times.map(toNyLocalIso),
                    y: all.values,
                    mode: 'lines',
                    line: { color: t.lineMuted, width: 2, shape: 'hv' },
                    opacity: 0.3,
                    showlegend: false
                });
            }
            for (const state of Object.keys(la)) {
                if (state === 'all' || !AA_STATE_COLORS[state]) continue;
                const s = la[state];
                if (!s || !s.times || !s.times.length) continue;
                traces.push({
                    type: 'scatter',
                    x: s.times.map(toNyLocalIso),
                    y: s.values,
                    mode: 'markers',
                    marker: { color: AA_STATE_COLORS[state], size: 10 },
                    opacity: 0.8,
                    name: state,
                    hovertemplate: '%{x|%H:%M:%S}<extra></extra>'
                });
            }
        }
        const layout = {
            title: { text: 'LeafAlly State', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                range: [-0.6, 8 - 0.8],
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor,
                tickvals: [0, 1, 2, 3, 4, 5, 6]
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    const RDBU_REV = [
        '#67001f',
        '#b2182b',
        '#d6604d',
        '#f4a582',
        '#fddbc7',
        '#f7f7f7',
        '#d1e5f0',
        '#92c5de',
        '#4393c3',
        '#2166ac',
        '#053061'
    ];

    function buildWeather(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const runs = payload.forecast_runs || [];
        const n = runs.length;
        const traces = [];
        runs.forEach((run, i) => {
            const isLatest = Boolean(run.is_latest);
            let color = 'red';
            if (!isLatest && n > 0) {
                color = RDBU_REV[Math.floor((i / n) * (RDBU_REV.length - 1))];
            }
            traces.push({
                type: 'scatter',
                x: (run.times_ms || []).map(toNyLocalIso),
                y: run.oat_f || [],
                mode: 'lines',
                line: { color, width: 2, shape: 'hv' },
                opacity: isLatest ? 1 : 0.2,
                showlegend: false,
                hovertemplate: '%{x|%H:%M:%S} | %{y}°F<extra></extra>'
            });
        });
        const layout = {
            title: { text: 'Weather Forecasts', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    function buildPrices(payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const t = theme(isDarkMode);
        const xRange = xRangeFromPayload(payload, toNyLocalIso);
        const lmp = payload.lmp_values || [];
        const dist = payload.dist_values || [];
        const times = payload.price_times_ms || [];
        const total = lmp.map((v, i) => v + (dist[i] || 0));
        const traces = [
            {
                type: 'scatter',
                x: times.map(toNyLocalIso),
                y: total,
                mode: 'lines',
                line: { color: t.lineMuted, shape: 'hv' },
                opacity: 0.8,
                showlegend: true,
                name: 'Total',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.2f} $/MWh<extra></extra>'
            },
            {
                type: 'scatter',
                x: times.map(toNyLocalIso),
                y: lmp,
                mode: 'lines',
                line: { color: t.lineMuted, dash: 'dot', shape: 'hv' },
                opacity: 0.4,
                showlegend: true,
                yaxis: 'y2',
                name: 'LMP',
                hovertemplate: '%{x|%H:%M:%S} | %{y:.2f} $/MWh<extra></extra>'
            }
        ];
        const layout = {
            title: { text: 'Price Forecast', x: 0.5, xanchor: 'center', font: { color: t.fontColor } },
            margin: { t: 30, b: 30 },
            plot_bgcolor: t.bg,
            paper_bgcolor: t.bg,
            font: { color: t.fontColor },
            hovermode: 'closest',
            xaxis: {
                range: xRange,
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                type: 'date'
            },
            yaxis: {
                title: 'Total price [$/MWh]',
                mirror: true,
                ticks: 'outside',
                showline: true,
                linecolor: t.fontColor,
                zeroline: false,
                showgrid: true,
                gridwidth: 1,
                gridcolor: t.gridColor
            },
            yaxis2: {
                title: 'LMP [$/MWh]',
                mirror: true,
                ticks: 'outside',
                zeroline: false,
                showline: true,
                linecolor: t.fontColor,
                showgrid: false,
                overlaying: 'y',
                side: 'right'
            },
            legend: {
                x: 0,
                y: 1,
                xanchor: 'left',
                yanchor: 'top',
                bgcolor: 'rgba(0, 0, 0, 0)'
            },
            shapes: latePersistenceShapes(payload.late_persistence_periods_ms, toNyLocalIso)
        };
        return { traces, layout };
    }

    const BUILDERS = {
        heatpump: buildHeatpump,
        distribution: buildDistribution,
        heatcalls: buildHeatcalls,
        zones: buildZones,
        buffer: buildBuffer,
        storage: buildStorage,
        top_state: buildTopState,
        ha_state: buildHaState,
        aa_state: buildAaState,
        weather: buildWeather,
        prices: buildPrices
    };

    function build(plotKind, payload, selectedChannels, isDarkMode, toNyLocalIso) {
        const fn = BUILDERS[plotKind];
        if (!fn) return null;
        return fn(payload, selectedChannels, isDarkMode, toNyLocalIso);
    }

    global.VisualizerPlots = { build, BUILDERS };
})(typeof window !== 'undefined' ? window : this);
