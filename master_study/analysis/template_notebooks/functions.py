from plotly.subplots import make_subplots
import numpy as np
import copy
import plotly.graph_objects as go
import logging
import json

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.info,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def return_bb_ho_dic(df_tw_b1, df_tw_b2, collider):
    # Find elements at extremities of each IP
    # IP1 : mqy.4l1.b1 to mqy.4r1.b1
    # IP2 : mqy.b5l2.b1 to mqy.b4r2.b1
    # IP5 : mqy.4l5.b1 to mqy.4r5.b1
    # IP8 : mqy.b4l8.b1 to mqy.b4r8.b1
    dic_bb_ho_IPs = {"lhcb1": {"sv": {}, "tw": {}}, "lhcb2": {"sv": {}, "tw": {}}}
    logging.info("Recomputing survey at each IP")
    for beam, df_tw in zip(["lhcb1", "lhcb2"], [df_tw_b1, df_tw_b2]):
        for ip, el_start, el_end in zip(
            ["ip1", "ip2", "ip5", "ip8"],
            ["mqy.4l1", "mqy.b4l2", "mqy.4l5", "mqy.b4l8"],
            ["mqy.4r1", "mqy.b4r2", "mqy.4r5", "mqy.b4r8"],
        ):
            # Change element name for current beam
            el_start = el_start + "." + beam[3:]
            el_end = el_end + "." + beam[3:]
            logging.info("Recompute survey start")
            # # Recompute survey from ip
            if beam == "lhcb1":
                df_sv = collider[beam].survey(element0=ip).to_pandas()
            else:
                df_sv = collider[beam].survey(element0=ip).reverse().to_pandas()
            logging.info("Recompute survey end")
            # Get twiss and sv between start and end element
            idx_element_start_tw = df_tw.index[df_tw.name == el_start].tolist()[0]
            idx_element_end_tw = df_tw.index[df_tw.name == el_end].tolist()[0]
            idx_element_start_sv = df_sv.index[df_sv.name == el_start].tolist()[0]
            idx_element_end_sv = df_sv.index[df_sv.name == el_end].tolist()[0]
            logging.info("Deepcopy start")
            # Get dataframe of elements between s_start and s_end
            dic_bb_ho_IPs[beam]["sv"][ip] = copy.deepcopy(
                df_sv.iloc[idx_element_start_sv : idx_element_end_sv + 1]
            )
            dic_bb_ho_IPs[beam]["tw"][ip] = copy.deepcopy(
                df_tw.iloc[idx_element_start_tw : idx_element_end_tw + 1]
            )
            logging.info("Deepcopy end")
    logging.info("Cleaning and harmonizing dataframes")
    # Delete all .b1 and .b2 from element names
    for ip in ["ip1", "ip2", "ip5", "ip8"]:
        dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "name"] = [
            el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["sv"][ip].name
        ]
        dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "name"] = [
            el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["sv"][ip].name
        ]
        dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "name"] = [
            el.replace(".b2", "").replace("b2_", "") for el in dic_bb_ho_IPs["lhcb2"]["tw"][ip].name
        ]
        dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "name"] = [
            el.replace(".b1", "").replace("b1_", "") for el in dic_bb_ho_IPs["lhcb1"]["tw"][ip].name
        ]

    for ip in ["ip1", "ip2", "ip5", "ip8"]:
        # Get intersection of names in twiss and survey
        s_intersection = (
            set(dic_bb_ho_IPs["lhcb2"]["sv"][ip].name)
            .intersection(set(dic_bb_ho_IPs["lhcb1"]["sv"][ip].name))
            .intersection(set(dic_bb_ho_IPs["lhcb2"]["tw"][ip].name))
            .intersection(set(dic_bb_ho_IPs["lhcb1"]["tw"][ip].name))
        )

        # Clean dataframes in both beams so that they are comparable
        for beam in ["lhcb1", "lhcb2"]:
            # Remove all rows whose name is not in both beams
            dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                dic_bb_ho_IPs[beam]["sv"][ip].name.isin(s_intersection)
            ]
            dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                dic_bb_ho_IPs[beam]["tw"][ip].name.isin(s_intersection)
            ]

            # Remove all elements whose name contains '..'
            for i in range(1, 6):
                dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                    ~dic_bb_ho_IPs[beam]["sv"][ip].name.str.endswith(f"..{i}")
                ]
                dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                    ~dic_bb_ho_IPs[beam]["tw"][ip].name.str.endswith(f"..{i}")
                ]

        # Center s around IP for beam 1
        dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"] = (
            dic_bb_ho_IPs["lhcb1"]["sv"][ip].loc[:, "s"]
            - dic_bb_ho_IPs["lhcb1"]["sv"][ip][
                dic_bb_ho_IPs["lhcb1"]["sv"][ip].name == ip
            ].s.to_numpy()
        )
        dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"] = (
            dic_bb_ho_IPs["lhcb1"]["tw"][ip].loc[:, "s"]
            - dic_bb_ho_IPs["lhcb1"]["tw"][ip][
                dic_bb_ho_IPs["lhcb1"]["tw"][ip].name == ip
            ].s.to_numpy()
        )

        # Set the s of beam 1 as reference for all dataframes
        dic_bb_ho_IPs["lhcb2"]["sv"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["sv"][ip].s.to_numpy()
        dic_bb_ho_IPs["lhcb2"]["tw"][ip].loc[:, "s"] = dic_bb_ho_IPs["lhcb1"]["tw"][ip].s.to_numpy()

        # Only keep bb_ho and bb_lr elements
        for beam in ["lhcb1", "lhcb2"]:
            dic_bb_ho_IPs[beam]["sv"][ip] = dic_bb_ho_IPs[beam]["sv"][ip][
                dic_bb_ho_IPs[beam]["sv"][ip].name.str.contains(f"bb_ho|bb_lr")
            ]
            dic_bb_ho_IPs[beam]["tw"][ip] = dic_bb_ho_IPs[beam]["tw"][ip][
                dic_bb_ho_IPs[beam]["tw"][ip].name.str.contains(f"bb_ho|bb_lr")
            ]
    logging.info("4")
    return dic_bb_ho_IPs


def return_separation_dic(dic_bb_ho_IPs, tw_b1, nemitt_x, nemitt_y, energy):
    dic_sep_IPs = {"v": {}, "h": {}}

    for idx, n_ip in enumerate([1, 2, 5, 8]):
        # s doesn't depend on plane
        s = dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].s

        # Horizontal separation
        x = abs(
            dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].x
            - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].x.to_numpy()
        )
        n_emitt = nemitt_x / energy
        sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].betx * n_emitt) ** 0.5
        xing = float(tw_b1.rows[f"ip{n_ip}"]["px"])
        beta = float(tw_b1.rows[f"ip{n_ip}"]["betx"])
        sep_survey = abs(
            dic_bb_ho_IPs["lhcb1"]["sv"][f"ip{n_ip}"].X
            - dic_bb_ho_IPs["lhcb2"]["sv"][f"ip{n_ip}"].X.to_numpy()
        )
        sep = xing * 2 * np.sqrt(beta / n_emitt)

        # Store everyting in dic
        dic_sep_IPs["h"][f"ip{n_ip}"] = {
            "s": s,
            "x": x,
            "sep": sep,
            "sep_survey": sep_survey,
            "sigma": sigma,
        }

        # Vertical separation
        x = abs(
            dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].y
            - dic_bb_ho_IPs["lhcb2"]["tw"][f"ip{n_ip}"].y.to_numpy()
        )
        n_emitt = nemitt_y / 7000
        sigma = (dic_bb_ho_IPs["lhcb1"]["tw"][f"ip{n_ip}"].bety * n_emitt) ** 0.5
        xing = abs(float(tw_b1.rows[f"ip{n_ip}"]["py"]))
        beta = float(tw_b1.rows[f"ip{n_ip}"]["bety"])
        sep_survey = 0
        sep = xing * 2 * np.sqrt(beta / n_emitt)

        # Store everyting in dic
        dic_sep_IPs["v"][f"ip{n_ip}"] = {
            "s": s,
            "x": x,
            "sep": sep,
            "sep_survey": sep_survey,
            "sigma": sigma,
        }
    return dic_sep_IPs


def return_plot_separation(dic_sep_IPs, title="Beam-beam separation at the different IPs"):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("IP 1", "IP 2", "IP 5", "IP 8"),
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
        ],
        horizontal_spacing=0.2,
    )
    for idx, n_ip in enumerate([1, 2, 5, 8]):
        s = dic_sep_IPs[f"ip{n_ip}"]["s"]
        x = dic_sep_IPs[f"ip{n_ip}"]["x"]
        sep = dic_sep_IPs[f"ip{n_ip}"]["sep"]
        sep_survey = dic_sep_IPs[f"ip{n_ip}"]["sep_survey"]
        sigma = dic_sep_IPs[f"ip{n_ip}"]["sigma"]

        # Do the plot
        fig.add_trace(
            go.Scatter(
                x=s,
                y=x + sep_survey,
                name="Separation at ip " + str(n_ip),
                legendgroup=" IP " + str(n_ip),
                mode="lines+markers",
                line=dict(color="coral", width=1),
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=s,
                y=(x + sep_survey) / sigma,
                name="Normalized separation at ip " + str(n_ip),
                legendgroup=" IP " + str(n_ip),
                mode="lines+markers",
                line=dict(color="cyan", width=1),
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=s,
                y=[sep] * len(s),
                name="Inner normalized separation at ip " + str(n_ip),
                legendgroup=" IP " + str(n_ip),
                mode="lines+text",
                textposition="top left",
                line=dict(color="white", width=1, dash="dash"),
                text=[""] * (len(s) - 1) + ["Inner normalized separation"],
            ),
            row=idx // 2 + 1,
            col=idx % 2 + 1,
            secondary_y=True,
        )

    for row in range(1, 3):
        for column in range(1, 3):
            fig.update_yaxes(
                title_text=r"$\textrm{B-B separation }[m]$",
                row=row,
                col=column,
                linecolor="coral",
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text=r"$\textrm{B-B separation }[\sigma]$",
                row=row,
                col=column,
                linecolor="cyan",
                secondary_y=True,
            )
            fig.update_xaxes(title_text=r"$s [m]$", row=row, col=column)

    # fig.update_yaxes(range=[0, 0.25], row = 1, col = 1, secondary_y= False)
    # Use white theme for graph, centered title
    fig.update_layout(
        template="plotly_dark",
        title="Beam-beam separation at the different IPs",
        title_x=0.5,
        # paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        showlegend=False,
    )

    return fig


def configure_beam_beam(collider, config_bb):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb:
        # Initialize filling pattern with empty values
        filling_pattern_cw = None
        filling_pattern_acw = None

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        if "pattern_fname" in config_bb["mask_with_filling_pattern"]:
            # Fill values if possible
            if config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None:
                fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
                with open(fname, "r") as fid:
                    filling = json.load(fid)
                filling_pattern_cw = filling["beam1"]
                filling_pattern_acw = filling["beam2"]

                # Only track bunch number if a filling pattern has been provided
                if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
                if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

                # Note that a bunch number must be provided if a filling pattern is provided
                # Apply filling pattern
                collider.apply_filling_pattern(
                    filling_pattern_cw=filling_pattern_cw,
                    filling_pattern_acw=filling_pattern_acw,
                    i_bunch_cw=i_bunch_cw,
                    i_bunch_acw=i_bunch_acw,
                )
    return collider


def compute_separation_and_return_both_planes(
    collider, energy, emittance, val_on_sep8h, config_BB, update_BB=True
):
    logging.info("Updating on_sep8h value")
    if collider.vars["on_sep8h"]._value != val_on_sep8h:
        collider.vars["on_sep8h"] = val_on_sep8h

        if update_BB:
            logging.info("Updating beam-beam")
            collider = configure_beam_beam(collider, config_BB)

    logging.info("Running all steps to compute separation")
    tw_b1 = collider.lhcb1.twiss()
    df_tw_b1 = tw_b1.to_pandas()
    df_tw_b2 = collider.lhcb2.twiss().reverse().to_pandas()
    logging.info("Making both lines comparable")
    dic_bb_ho_IPs = return_bb_ho_dic(df_tw_b1, df_tw_b2, collider)
    logging.info("Computing separation")
    dic_sep_IPs = return_separation_dic(dic_bb_ho_IPs, tw_b1, emittance, emittance, energy)
    logging.info("Computing plot")
    fig_h = return_plot_separation(
        dic_sep_IPs["h"], title="Horizontal beam-beam separation at the different IPs"
    )
    fig_v = return_plot_separation(
        dic_sep_IPs["v"], title="Vertical beam-beam separation at the different IPs"
    )
    logging.info("Finished")
    return fig_h, fig_v
