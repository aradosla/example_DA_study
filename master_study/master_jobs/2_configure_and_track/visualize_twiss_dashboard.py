#################### Imports ####################

# Import standard libraries
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash
import numpy as np
import base64
import xtrack as xt
import io
import json

# Import functions
import dashboard_functions


#################### Build CSS ####################
dashboard_functions.build_CSS()

#################### Load global variables ####################

path_configuration = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma5_1p4_eol_bunch_scan/base_collider/xtrack_0002/config.yaml"

# Get collider variables
collider, tw_b1, df_sv_b1, df_tw_b1, tw_b2, df_sv_b2, df_tw_b2, df_elements_corrected = (
    dashboard_functions.return_all_loaded_variables(
        collider_path="/afs/cern.ch/work/c/cdroin/private/comparison_pymask_xmask/xmask/xsuite_lines/collider_03_tuned_and_leveled_bb_off.json"
    )
)

# Get corresponding data tables
table_sv_b1 = dashboard_functions.return_data_table(df_sv_b1, "id-df-sv-b1", twiss=False)
table_tw_b1 = dashboard_functions.return_data_table(df_tw_b1, "id-df-tw-b1", twiss=True)
table_sv_b2 = dashboard_functions.return_data_table(df_sv_b2, "id-df-sv-b2", twiss=False)
table_tw_b2 = dashboard_functions.return_data_table(df_tw_b2, "id-df-tw-b2", twiss=True)

# Get filling scheme
path_filling_scheme = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/8b4e_1972b_1960_1178_1886_224bpi_12inj_800ns_bs200ns.json"

with open(path_filling_scheme) as fid:
    filling_scheme = json.load(fid)

array_b1 = np.array(filling_scheme["beam1"])
array_b2 = np.array(filling_scheme["beam2"])

#################### App ####################
app = Dash(
    __name__,
    title="Dashboard for current simulation",
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
    ],
    # suppress_callback_exceptions=True,
)
server = app.server

#################### App Layout ####################


def return_header_layout():
    def create_header_link(icon, href, size=22, color="cyan"):
        return dmc.Anchor(
            dmc.ThemeIcon(
                DashIconify(
                    icon=icon,
                    width=size,
                ),
                variant="outline",
                radius=30,
                size=36,
                color=color,
            ),
            href=href,
            target="_blank",
        )

    header = dmc.Header(
        height=70,
        fixed=True,
        px=25,
        children=[
            dmc.Stack(
                justify="center",
                style={"height": 70, "width": "100%"},
                children=dmc.Group(
                    position="apart",
                    style={"width": "100%"},
                    children=[
                        dmc.Text(
                            "Simulation dashboard",
                            size=30,
                            color="cyan",
                        ),
                        dmc.SegmentedControl(
                            id="tab-titles",
                            value="display-configuration",
                            data=[
                                {"value": "display-configuration", "label": "Configuration"},
                                {"value": "display-twiss", "label": "Twiss tables"},
                                {"value": "display-scheme", "label": "Filling scheme"},
                                {"value": "display-optics", "label": "Optics"},
                                {"value": "display-survey", "label": "Survey"},
                            ],
                            # color="cyan",
                            style={"margin-right": "10%"},
                        ),
                        dmc.Group(
                            position="right",
                            spacing="xl",
                            children=[
                                create_header_link(
                                    "radix-icons:github-logo",
                                    "https://github.com/ColasDroin/example_DA_study",
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ],
    )
    return header


def return_configuration_layout(path_configuration):
    # Load configuration file
    with open(path_configuration, "r") as file:
        configuration_str = file.read()

    configuration_layout = dmc.Center(
        dmc.Prism(
            language="yaml",
            children=configuration_str,
            style={"height": "90vh", "overflowY": "auto", "width": "80%"},
        )
    )

    return configuration_layout


def return_tables_layout():
    layout = html.Div(
        children=[
            dmc.Center(
                dmc.Alert(
                    (
                        "The datatables are slow as they are"
                        " heavy to download from the server. If"
                        " we want to keep this feature, I will"
                        " try to implement a lazy loading,"
                        " sorting and filtering in the backend"
                        " to speed things up."
                    ),
                    title="Alert!",
                    style={
                        "width": "70%",
                        "margin-top": "10px",
                    },
                ),
            ),
            dmc.Center(
                dmc.SegmentedControl(
                    id="segmented-data-table",
                    data=[
                        "Twiss table beam 1",
                        "Survey table beam 1",
                        "Twiss table beam 2",
                        "Survey table beam 2",
                    ],
                    radius="md",
                    mt=10,
                    value="Twiss table beam 1",
                    color="cyan",
                ),
            ),
            html.Div(id="placeholder-data-table"),
        ],
        style={"width": "100%", "margin": "auto"},
    )
    return layout


def return_survey_layout():
    survey_layout = html.Div(
        children=[
            dmc.Center(
                dmc.Stack(
                    children=[
                        dmc.Center(
                            children=[
                                dmc.Group(
                                    children=[
                                        dmc.Text("Sectors to display: "),
                                        dmc.ChipGroup(
                                            [
                                                dmc.Chip(
                                                    x,
                                                    value=x,
                                                    variant="outline",
                                                    color="cyan",
                                                )
                                                for x in ["8-2", "2-4", "4-6", "6-8"]
                                            ],
                                            id="chips-ip",
                                            value=["4-6"],
                                            multiple=True,
                                            mb=0,
                                        ),
                                    ],
                                    pt=10,
                                ),
                            ],
                        ),
                        dcc.Loading(
                            children=dcc.Graph(
                                id="LHC-layout",
                                mathjax=True,
                                config={
                                    "displayModeBar": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                    "displaylogo": False,
                                },
                                style={"height": "90vh", "width": "100%", "margin": "auto"},
                            ),
                            type="circle",
                        ),
                    ],
                    style={"width": "100%", "margin": "auto"},
                )
            ),
            dmc.Drawer(
                title="Element information",
                id="drawer-magnets",
                padding="md",
                transition="rotate-left",
                transitionDuration=20,
                zIndex=10000,
                transitionTimingFunction="ease",
                children=dmc.Card(
                    children=[
                        dmc.Group(
                            [
                                dmc.Text(
                                    id="title-element",
                                    children="Element",
                                    weight=500,
                                ),
                                dmc.Badge(
                                    id="type-element",
                                    children="Dipole",
                                    color="blue",
                                    variant="light",
                                ),
                            ],
                            position="apart",
                            mt="md",
                            mb="xs",
                        ),
                        html.Div(
                            id="text-element",
                            children=[
                                dmc.Text(
                                    id="initial-text",
                                    children=(
                                        "Please click on a multipole or an"
                                        " interaction point to get the"
                                        " corresponding knob information."
                                    ),
                                    size="sm",
                                    color="dimmed",
                                ),
                            ],
                        ),
                    ],
                    withBorder=True,
                    shadow="sm",
                    radius="md",
                    style={"width": "100%"},
                ),
            ),
        ],
        style={"width": "100%", "margin": "auto"},
    )
    return survey_layout


def return_filling_scheme_layout():
    scheme_layout = dmc.Stack(
        children=[
            dmc.Center(
                dmc.Alert(
                    (
                        "I may add a plot displaying the number of long-ranges and head-on"
                        " interaction for each bunch is it's deemed relevant."
                    ),
                    title="Alert!",
                    style={"width": "70%", "margin-top": "10px"},
                ),
            ),
            dcc.Graph(
                id="filling-scheme-graph",
                mathjax=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": True,
                    "responsive": True,
                    "displaylogo": False,
                },
                figure=dashboard_functions.return_plot_filling_scheme(array_b1, array_b2),
                style={"height": "30vh", "width": "100%", "margin": "10 auto"},
            ),
        ]
    )
    return scheme_layout


def return_optics_layout():
    optics_layout = dmc.Center(
        dcc.Graph(
            id="LHC-2D-near-IP",
            mathjax=True,
            config={
                "displayModeBar": False,
                "scrollZoom": True,
                "responsive": True,
                "displaylogo": False,
            },
            figure=dashboard_functions.return_plot_optics(
                tw_b1, tw_b2, df_sv_b1, df_elements_corrected
            ),
            style={"height": "90vh", "width": "100%", "margin": "auto"},
        ),
    )
    return optics_layout


layout = html.Div(
    style={"width": "90%", "margin": "auto"},
    children=[
        # Interval for the logging handler
        # dcc.Interval(id="interval1", interval=5 * 1000, n_intervals=0),
        return_header_layout(),
        dmc.Center(
            children=[
                html.Div(
                    id="main-div",
                    style={"width": "100%", "margin": "auto"},
                    children=[
                        html.Div(id="placeholder-tabs"),
                        # dmc.TabsPanel(
                        #     children=return_configuration_layout(path_configuration),
                        #     value="display-configuration",
                        # ),
                        # dmc.TabsPanel(
                        #     children=return_tables_layout(),
                        #     value="display-twiss",
                        #     style={"height": "90vh", "width": "80%", "margin": "auto"},
                        # ),
                        # dmc.TabsPanel(
                        #     children=return_filling_scheme_layout(),
                        #     value="display-scheme",
                        # ),
                        # dmc.TabsPanel(children=return_optics_layout(), value="display-optics"),
                        # dmc.TabsPanel(
                        #     children=return_survey_layout(),
                        #     value="display-survey",
                        # ),
                    ],
                ),
            ],
            style={"margin-top": "75px"},
        ),
    ],
)

# Dark theme
layout = dmc.MantineProvider(
    withGlobalStyles=True,
    theme={"colorScheme": "dark"},
    children=layout,
)

app.layout = layout


#################### App Callbacks ####################


@app.callback(Output("placeholder-tabs", "children"), Input("tab-titles", "value"))
def select_tab(value):
    match value:
        case "display-configuration":
            return return_configuration_layout(path_configuration)
        case "display-twiss":
            return return_tables_layout()
        case "display-scheme":
            return return_filling_scheme_layout()
        case "display-optics":
            return return_optics_layout()
        case "display-survey":
            return return_survey_layout()
        case _:
            return return_configuration_layout(path_configuration)


@app.callback(Output("placeholder-data-table", "children"), Input("segmented-data-table", "value"))
def select_data_table(value):
    match value:
        case "Twiss table beam 1":
            return table_tw_b1
        case "Survey table beam 1":
            return table_sv_b1
        case "Twiss table beam 2":
            return table_tw_b2
        case "Survey table beam 2":
            return table_sv_b2
        case _:
            return table_tw_b1


@app.callback(
    Output("LHC-layout", "figure"),
    Input("chips-ip", "value"),
)
def update_graph_LHC_layout(l_values):
    l_indices_to_keep = []
    for val in l_values:
        str_ind_1, str_ind_2 = val.split("-")
        # Get indices of elements to keep (# ! implemented only for beam 1)
        l_indices_to_keep.extend(
            dashboard_functions.get_indices_of_interest(
                df_tw_b1, "ip" + str_ind_1, "ip" + str_ind_2
            )
        )

    fig = dashboard_functions.return_plot_lattice_with_tracking(
        df_sv_b1,
        df_elements_corrected,
        df_tw_b1,
        df_sv_2=df_sv_b2,
        df_tw_2=df_tw_b2,
        l_indices_to_keep=l_indices_to_keep,
    )

    return fig


@app.callback(
    Output("text-element", "children"),
    Output("title-element", "children"),
    Output("type-element", "children"),
    Output("drawer-magnets", "opened"),
    Input("LHC-layout", "clickData"),
    prevent_initial_call=False,
)
def update_text_graph_LHC_2D(clickData):
    if clickData is not None:
        if "customdata" in clickData["points"][0]:
            name = clickData["points"][0]["customdata"]
            if name.startswith("mb"):
                type_text = "Dipole"
                try:
                    set_var = collider.lhcb1.element_refs[name].knl[0]._expr._get_dependencies()
                except:
                    set_var = (
                        collider.lhcb1.element_refs[name + "..1"].knl[0]._expr._get_dependencies()
                    )
            elif name.startswith("mq"):
                type_text = "Quadrupole"
                try:
                    set_var = collider.lhcb1.element_refs[name].knl[1]._expr._get_dependencies()
                except:
                    set_var = (
                        collider.lhcb1.element_refs[name + "..1"].knl[1]._expr._get_dependencies()
                    )
            elif name.startswith("ms"):
                type_text = "Sextupole"
                try:
                    set_var = collider.lhcb1.element_refs[name].knl[2]._expr._get_dependencies()
                except:
                    set_var = (
                        collider.lhcb1.element_refs[name + "..1"].knl[2]._expr._get_dependencies()
                    )
            elif name.startswith("mo"):
                type_text = "Octupole"
                try:
                    set_var = collider.lhcb1.element_refs[name].knl[3]._expr._get_dependencies()
                except:
                    set_var = (
                        collider.lhcb1.element_refs[name + "..1"].knl[3]._expr._get_dependencies()
                    )

            text = []
            for var in set_var:
                name_var = str(var).split("'")[1]
                val = collider.lhcb1.vars[name_var]._get_value()
                expr = collider.lhcb1.vars[name_var]._expr
                if expr is not None:
                    dependencies = collider.lhcb1.vars[name_var]._expr._get_dependencies()
                else:
                    dependencies = "No dependencies"
                    expr = "No expression"
                targets = collider.lhcb1.vars[name_var]._find_dependant_targets()

                text.append(dmc.Text("Name: ", weight=500))
                text.append(dmc.Text(name_var, size="sm"))
                text.append(dmc.Text("Element value: ", weight=500))
                text.append(dmc.Text(str(val), size="sm"))
                text.append(dmc.Text("Expression: ", weight=500))
                text.append(dmc.Text(str(expr), size="sm"))
                text.append(dmc.Text("Dependencies: ", weight=500))
                text.append(dmc.Text(str(dependencies), size="sm"))
                text.append(dmc.Text("Targets: ", weight=500))
                if len(targets) > 10:
                    text.append(
                        dmc.Text(str(targets[:10]), size="sm"),
                    )
                    text.append(dmc.Text("...", size="sm"))
                else:
                    text.append(dmc.Text(str(targets), size="sm"))

            return text, name, type_text, True

    return (
        dmc.Text("Please click on a multipole to get the corresponding knob information."),
        dmc.Text("Click !"),
        dmc.Text("Undefined type"),
        False,
    )


#################### Launch app ####################
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)


# Run with gunicorn app:server -b :8000
# Run silently with nohup gunicorn app:server -b :8000 &
# Kill with pkill gunicorn
