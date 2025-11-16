from typing import Any

import pandas as pd
import panel as pn

from src.core.data_sources import DataSource, LocalFileDataSource

try:
    from src.core.data_sources import SnowflakeDataSource
except ImportError:
    SnowflakeDataSource = None

try:
    from src.core.data_sources import AWSDataSource
except ImportError:
    AWSDataSource = None
from src.core.data_sources.base import DataSourceConfig


class DataManagementPanel:
    def __init__(self) -> None:
        self.current_datasource: DataSource | None = None
        self.current_data: pd.DataFrame | None = None
        self.data_updated_callback = None

        # UI Components - only include available data sources
        available_sources = ["Local Files"]
        if SnowflakeDataSource is not None:
            available_sources.append("Snowflake")
        if AWSDataSource is not None:
            available_sources.append("AWS S3")

        self.datasource_type_select = pn.widgets.Select(
            name="Data Source Type", options=available_sources, value="Local Files"
        )

        self.connection_inputs = pn.Column()
        self.connect_button = pn.widgets.Button(
            name="Connect", button_type="primary", width=150
        )

        self.table_select = pn.widgets.Select(
            name="Available Tables/Files", options=[], disabled=True
        )

        self.load_button = pn.widgets.Button(
            name="Load Data", button_type="success", disabled=True, width=150
        )

        self.data_preview = pn.pane.DataFrame(pd.DataFrame({}), width=800, height=400)

        self.data_profile = pn.pane.JSON({}, theme="light")

        # Status indicators
        self.connection_status = pn.pane.HTML(
            "<span style='color: red;'>●</span> Not Connected"
        )

        # Set up callbacks
        self._setup_callbacks()

        # Create the main panel
        self.panel = self._create_panel()

    def _setup_callbacks(self) -> None:
        self.datasource_type_select.param.watch(
            self._on_datasource_type_change, "value"
        )
        self.connect_button.on_click(self._on_connect)
        self.load_button.on_click(self._on_load_data)
        self.table_select.param.watch(self._on_table_select, "value")

    def _create_panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## Data Source Configuration"),
            pn.Row(
                pn.Column(
                    self.datasource_type_select,
                    self.connection_inputs,
                    pn.Row(self.connect_button, self.connection_status),
                    width=400,
                ),
                pn.Column(self.table_select, self.load_button, width=300),
            ),
            pn.pane.Markdown("## Data Preview"),
            self.data_preview,
            pn.pane.Markdown("## Data Profile"),
            self.data_profile,
        )

    def _on_datasource_type_change(self, event: Any) -> None:
        self.connection_inputs.clear()

        if event.new == "Local Files":
            self.connection_inputs.extend(
                [
                    pn.widgets.TextInput(
                        name="Base Path",
                        value="./data",
                        placeholder="Path to data directory",
                    )
                ]
            )
        elif event.new == "Snowflake":
            self.connection_inputs.extend(
                [
                    pn.widgets.TextInput(name="User", placeholder="Snowflake username"),
                    pn.widgets.PasswordInput(name="Password", placeholder="Password"),
                    pn.widgets.TextInput(
                        name="Account", placeholder="Account identifier"
                    ),
                    pn.widgets.TextInput(
                        name="Warehouse", placeholder="Warehouse name"
                    ),
                    pn.widgets.TextInput(name="Database", placeholder="Database name"),
                    pn.widgets.TextInput(name="Schema", placeholder="Schema name"),
                ]
            )
        elif event.new == "AWS S3":
            self.connection_inputs.extend(
                [
                    pn.widgets.TextInput(
                        name="Access Key ID", placeholder="AWS Access Key"
                    ),
                    pn.widgets.PasswordInput(
                        name="Secret Access Key", placeholder="AWS Secret Key"
                    ),
                    pn.widgets.TextInput(
                        name="Region", value="us-east-1", placeholder="AWS Region"
                    ),
                    pn.widgets.TextInput(
                        name="Bucket Name", placeholder="S3 Bucket Name"
                    ),
                ]
            )

    def _get_connection_params(self) -> dict[str, Any]:
        params = {}
        for widget in self.connection_inputs:
            if hasattr(widget, "name") and hasattr(widget, "value"):
                key = widget.name.lower().replace(" ", "_")
                params[key] = widget.value
        return params

    def _on_connect(self, event: Any) -> None:
        try:
            params = self._get_connection_params()

            config = DataSourceConfig(
                name=f"{self.datasource_type_select.value}_connection",
                source_type=self.datasource_type_select.value.lower().replace(" ", "_"),
                connection_params=params,
            )

            # Create appropriate datasource
            if self.datasource_type_select.value == "Local Files":
                self.current_datasource = LocalFileDataSource(config)
            elif (
                self.datasource_type_select.value == "Snowflake"
                and SnowflakeDataSource is not None
            ):
                self.current_datasource = SnowflakeDataSource(config)
            elif (
                self.datasource_type_select.value == "AWS S3"
                and AWSDataSource is not None
            ):
                self.current_datasource = AWSDataSource(config)
            else:
                self._raise_unsupported_datasource_error(
                    self.datasource_type_select.value
                )

            # Test connection
            if self.current_datasource.test_connection():
                self.connection_status.object = (
                    "<span style='color: green;'>●</span> Connected"
                )

                # Load available tables
                tables = self.current_datasource.list_tables()
                self.table_select.options = tables
                self.table_select.disabled = False

            else:
                self.connection_status.object = (
                    "<span style='color: red;'>●</span> Connection Failed"
                )

        except Exception as e:
            self.connection_status.object = (
                f"<span style='color: red;'>●</span> Error: {e!s}"
            )

    def _raise_unsupported_datasource_error(self, datasource_type: str) -> None:
        """Raise a ValueError for unsupported or unavailable datasources."""
        msg = f"Unsupported or unavailable datasource: {datasource_type}"
        raise ValueError(msg)

    def _on_table_select(self, event: Any) -> None:
        if event.new:
            self.load_button.disabled = False

    def _on_load_data(self, event: Any) -> None:
        if not self.current_datasource or not self.table_select.value:
            print("No datasource or table selected")
            return
        # Try to update data preview
        try:
            # Load data preview
            if isinstance(self.current_datasource, LocalFileDataSource):
                self.current_data = self.current_datasource.load_data(
                    self.table_select.value
                )
            else:
                self.current_data = self.current_datasource.load_data(
                    self.table_select.value
                )
            # Update preview (show first 100 rows)
            # logger.debug(f"Loaded data from {self.table_select.value}")  # noqa: ERA001
            preview_data = self.current_data.head(100).iloc[:, :]
            # logger.debug(f"Data preview:\n{preview_data}")  # noqa: ERA001
            # logger.debug(preview_data.info())  # noqa: ERA001
            # If no data is available in the preview, show a message instead of an empty table
            if preview_data.empty:
                preview_data = pd.DataFrame(
                    {
                        "Message": [
                            "Data preview table is empty. Check the data configuration, query or filters."
                        ]
                    }
                )
            self.data_preview.object = preview_data

        except Exception as e:
            # logger.error(f"Error updating data preview: {e!s}")  # noqa: ERA001
            self.data_preview.object = pd.DataFrame(
                {"Error": [f"Failed to load data preview: {e!s}"]}
            )
        # Try to update data profile
        try:
            profile = self.current_datasource.get_data_profile(self.current_data)
            self.data_profile.object = profile

            # Notify other panels about data update
            if self.data_updated_callback:
                self.data_updated_callback(self.current_data)

        except ValueError as e:
            self.data_profile.object = pd.DataFrame(
                {"Error": [f"Failed to load data profile: {e!s}"]}
            )
