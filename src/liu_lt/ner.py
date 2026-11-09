import ipywidgets as widgets
from IPython.display import display

from scipy.stats import hmean
import spacy
from spacy import displacy
from tabulate import tabulate

nlp = spacy.blank("en")


def display_ner(text, ents, jupyter=True):
    doc = nlp(text)
    ents = [
        doc.char_span(ent["start"], ent["end"], label=ent["entity_group"]) for ent in ents
    ]
    doc.ents = [ent for ent in ents if ent is not None]
    if jupyter:
        displacy.render(doc, style="ent")
    else:
        return displacy.render(doc, style="ent", jupyter=False)


def evaluation_report(counts):
    report = []
    labels = sorted(counts["true"].keys() | counts["pred"].keys())

    for label in labels:
        # Compute precision
        if counts["pred"][label] > 0:
            precision = counts["correct"][label] / counts["pred"][label]
        else:
            precision = 0
        # Compute recall
        if counts["true"][label] > 0:
            recall = counts["correct"][label] / counts["true"][label]
        else:
            recall = 0
        # Compute F1-score, using scipy's hmean (harmonic mean) function
        fscore = float(hmean([precision, recall]))
        report.append([label, precision, recall, fscore])

    # Compute average over all labels
    precision = counts["correct"].total() / counts["pred"].total()
    recall = counts["correct"].total() / counts["true"].total()
    fscore = float(hmean([precision, recall]))
    report.append(["Average", precision, recall, fscore])

    return tabulate(
        report,
        tablefmt="html",
        floatfmt=".2f",
        headers=["Entity label", "Precision", "Recall", "F1-score"],
    )


def interactive_ner_browser(data):
    # Courtesy of Claude Sonnet 4.5...

    # Create widgets
    output_true = widgets.HTML()
    output_pred = widgets.HTML()
    prev_button = widgets.Button(
        description="", icon="caret-left", layout=widgets.Layout(width="40px")
    )
    next_button = widgets.Button(
        description="", icon="caret-right", layout=widgets.Layout(width="40px")
    )
    page_input = widgets.IntText(
        value=1,
        min=1,
        max=len(data),
        step=1,
        description="Example:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="160px"),
    )
    page_label = widgets.Label(value=f"of {len(data)}")

    # State to track current index
    current_index = [0]  # Using a list so we can modify it in nested functions

    def update_display():
        """Update the output and label"""
        idx = current_index[0]
        output_true.value = display_ner(
            data[idx]["text"], data[idx]["ents"], jupyter=False
        )
        output_pred.value = display_ner(
            data[idx]["text"], data[idx]["pred"], jupyter=False
        )

        # Update page input (using 1-based indexing for user)
        page_input.value = current_index[0] + 1

        # Enable/disable buttons based on position
        prev_button.disabled = current_index[0] == 0
        next_button.disabled = current_index[0] == len(data) - 1

    def on_prev_clicked(b):
        if current_index[0] > 0:
            current_index[0] -= 1
            update_display()

    def on_next_clicked(b):
        if current_index[0] < len(data) - 1:
            current_index[0] += 1
            update_display()

    def on_page_change(change):
        """Handle manual page input changes"""
        # Convert from 1-based to 0-based indexing
        new_index = change["new"] - 1
        if 0 <= new_index < len(data) and new_index != current_index[0]:
            current_index[0] = new_index
            update_display()

    # Attach event handlers
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)
    page_input.observe(on_page_change, names="value")

    # Create layout
    button_box = widgets.HBox([prev_button, page_input, page_label, next_button])
    container = widgets.VBox(
        [
            button_box,
            widgets.HTML("<b>True:</b>"),
            output_true,
            widgets.HTML("<b>Predicted:</b>"),
            output_pred,
        ]
    )

    # Initial display
    update_display()

    # Display the widget
    display(container)
