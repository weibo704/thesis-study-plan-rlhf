"""Generate synthetic source CSVs that mirror the real schemas and defects.

These are NOT the real datasets. They exist so the pipeline can be exercised
end to end before Kaggle credentials are configured, and so the unit tests
have deterministic input. The column names, sentinel values, duration
formats and encoding damage all reproduce what was verified on the live
Kaggle pages on 2026-09-20:

  * coursera2021 : 7 cols, "Not Calibrated" in Difficulty Level AND Course
                   Rating, space-joined undelimited Skills blob, cp1252
                   mojibake on non-ASCII institution names
  * coursera2025 : 9 cols, no URL, no description, bucketed Duration
                   strings, "Mixed" level, Specialization rows
  * edx          : 16 cols, syllabus text, "5 hours per week" x "6 weeks",
                   non-English rows, course_type conflating pace with
                   MicroMasters membership
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

COURSERA2021_HEADER = [
    "Course Name", "University", "Difficulty Level", "Course Rating",
    "Course URL", "Course Description", "Skills",
]

COURSERA2021_ROWS = [
    ["Machine Learning Foundations", "Stanford University", "Beginner", "4.8",
     "https://www.coursera.org/learn/machine-learning-foundations",
     "Learn supervised learning, linear regression and model evaluation from scratch.",
     "machine learning supervised learning linear regression model evaluation python"],
    ["Deep Learning with PyTorch", "DeepLearning.AI", "Intermediate", "4.7",
     "https://www.coursera.org/learn/deep-learning-pytorch",
     "Build and train neural networks using PyTorch, covering CNNs and transfer learning.",
     "deep learning pytorch neural networks convolutional networks transfer learning"],
    # Duplicate of the row above under a different title: exercises dedup.
    ["Deep Learning With PyTorch", "DeepLearning.AI", "Intermediate", "4.7",
     "https://www.coursera.org/learn/deep-learning-pytorch",
     "Build and train neural networks using PyTorch, covering CNNs and transfer learning.",
     "deep learning pytorch neural networks"],
    # Missing description + both sentinels: exercises null handling.
    ["Advanced SQL for Data Engineers", "University of California", "Not Calibrated",
     "Not Calibrated", "https://www.coursera.org/learn/advanced-sql", "",
     "sql database query optimization indexing"],
    # Mojibake in the institution name, as produced by the real export.
    ["Silicon Thin Film Solar Cells", "\xc3\x89cole Polytechnique", "Advanced", "4.1",
     "https://www.coursera.org/learn/silicon-thin-film-solar-cells",
     "A general presentation of solar cells based on silicon thin films.",
     "chemistry physics Solar Energy film lambda calculus electronics"],
    # Out-of-scope course that carries a false-positive AI-ish skill tag.
    ["Write A Feature Length Screenplay", "Michigan State University", "Beginner", "4.8",
     "https://www.coursera.org/learn/write-a-feature-length-screenplay",
     "Write a complete feature-length screenplay for film or television.",
     "Drama Comedy peering screenwriting film Document Review dialogue creative writing"],
    ["Docker and Kubernetes for Developers", "IBM", "Intermediate", "4.6",
     "https://www.coursera.org/learn/docker-kubernetes",
     "Containerize applications with Docker and orchestrate them on Kubernetes.",
     "docker kubernetes containers deployment devops"],
    # A container that must be excluded from the arm pool.
    ["Applied Data Science Specialization", "IBM", "Beginner", "4.6",
     "https://www.coursera.org/specializations/applied-data-science",
     "A multi-course programme covering the data science workflow end to end.",
     "data science python pandas visualization"],
    ["Natural Language Processing with Transformers", "DeepLearning.AI", "Advanced", "4.9",
     "https://www.coursera.org/learn/nlp-transformers",
     "Attention, transformer architectures and fine-tuning for text tasks.",
     "nlp transformers attention fine tuning language models"],
    ["Introduction to Python Programming", "University of Michigan", "Beginner", "4.8",
     "https://www.coursera.org/learn/python-intro",
     "A no prior experience introduction to Python, variables, loops and functions.",
     "python programming variables loops functions"],
]

COURSERA2025_HEADER = [
    "Subject", "Title", "Institution", "Learning Product", "Level",
    "Duration", "Gained Skills", "Rate", "Reviews",
]

COURSERA2025_ROWS = [
    ["Data Science", "Machine Learning Foundations", "Stanford University", "Course",
     "Beginner", "1 - 3 Months", "Machine Learning, Supervised Learning, Python", "4.8", "12,430"],
    ["Computer Science", "Deep Learning with PyTorch", "DeepLearning.AI", "Course",
     "Intermediate", "1 - 4 Weeks", "Deep Learning, PyTorch, Neural Networks", "4.7", "8,201"],
    ["Computer Science", "Advanced SQL for Data Engineers", "University of California",
     "Course", "Advanced", "Less Than 2 Hours", "SQL, Databases, Query Optimization", "4.5", "930"],
    ["Information Technology", "Docker and Kubernetes for Developers", "IBM", "Course",
     "Intermediate", "1 - 3 Months", "Docker, Kubernetes, DevOps", "4.6", "3,110"],
    ["Data Science", "Applied Data Science Specialization", "IBM", "Specialization",
     "Mixed", "3 - 6 Months", "Data Science, Python, Pandas", "4.6", "44,102"],
    # Donor-only row: must never create a course.
    ["Business", "Financial Markets", "Yale University", "Course", "Beginner",
     "1 - 3 Months", "Finance, Investment", "4.8", "27,004"],
]

EDX_HEADER = [
    "title", "summary", "n_enrolled", "course_type", "institution", "instructors",
    "Level", "subject", "language", "subtitles", "course_effort", "course_length",
    "price", "course_description", "course_syllabus", "course_url",
]

EDX_ROWS = [
    ["CS50's Introduction to Computer Science",
     "An introduction to the intellectual enterprises of computer science.",
     "2,442,271", "Self-paced on your time", "Harvard University",
     "David J. Malan-Doug Lloyd", "Introductory", "Computer Science", "English",
     "English", "6-18 hours per week", "12 weeks", "FREE-Add a Verified Certificate for $199",
     "An entry-level course teaching students how to think algorithmically and solve problems efficiently.",
     "Week 0 Scratch; Week 1 C; Week 2 Arrays; Week 3 Algorithms. No prior experience required.",
     "https://www.edx.org/course/cs50s-introduction-to-computer-science"],
    ["Machine Learning with Python",
     "Learn the fundamentals of machine learning with Python.",
     "129,555", "Instructor-led on a course schedule", "IBM",
     "Saeed Aghabozorgi", "Intermediate", "Computer Science", "English", "English",
     "4 hours per week", "6 weeks", "$99",
     "Covers regression, classification and clustering using scikit-learn.",
     "Prerequisites: basic Python programming and introductory statistics. Week 1 regression.",
     "https://www.edx.org/course/machine-learning-with-python"],
    ["Data Science: R Basics", "This course is part of a MicroMasters Program",
     "81,140", "This course is part of a MicroMasters Program", "Harvard University",
     "Rafael Irizarry", "Introductory", "Data Analysis & Statistics", "English", "English",
     "2 hours per week", "8 weeks", "FREE",
     "Build a foundation in R and learn to wrangle, analyze and visualize data.",
     "Requires no prior programming knowledge. Week 1 R basics.",
     "https://www.edx.org/course/data-science-r-basics"],
    # Non-English row: must be filtered from the arm pool.
    ["Introducci\xc3\xb3n a la Programaci\xc3\xb3n", "Aprende a programar desde cero.",
     "45,000", "Self-paced on your time", "Universitat Polit\xc3\xa8cnica de Valencia",
     "Equipo UPV", "Introductory", "Computer Science", "Espa\xc3\xb1ol", "Espa\xc3\xb1ol",
     "5 hours per week", "4 weeks", "FREE",
     "Curso introductorio de programaci\xc3\xb3n.", "Semana 1 variables.",
     "https://www.edx.org/course/introduccion-a-la-programacion"],
    # No duration fields at all: effort_hours_total must stay null.
    ["Introduction to Cloud Computing", "Cloud fundamentals for developers.",
     "12,000", "Self-paced on your time", "IBM", "Rav Ahuja", "Introductory",
     "Computer Science", "English", "English", "", "", "FREE",
     "An overview of cloud service models, deployment models and providers.",
     "", "https://www.edx.org/course/introduction-to-cloud-computing"],
]


COURSERA_HOURS_HEADER = [
    "Course Title", "Rating", "Level", "Duration", "Schedule", "Review",
    "What you will learn", "Skill gain", "Modules", "Instructor", "Offered By",
    "Keyword", "Course Url",
]

# Slug-joinable donor carrying REAL effort hours. Slugs deliberately overlap
# the coursera2021 fixture so the exact-slug join is exercised.
COURSERA_HOURS_ROWS = [
    ["Machine Learning Foundations", "4.8", "Beginner level", "20 hours (approximately)",
     "Flexible schedule", "12,430 reviews", "", "['Machine Learning', 'Supervised Learning']",
     "Intro", "Andrew Ng", "Stanford University", "ml",
     "https://www.coursera.org/learn/machine-learning-foundations"],
    ["Deep Learning with PyTorch", "4.7", "Intermediate level",
     "Approx. 34 hours to complete", "Flexible schedule", "8,201 reviews", "",
     "['Deep Learning', 'PyTorch', 'Neural Networks']", "Intro", "Joseph Santarcangelo",
     "DeepLearning.AI", "dl", "https://www.coursera.org/learn/deep-learning-pytorch"],
    ["Docker and Kubernetes for Developers", "4.6", "Intermediate level",
     "Approx. 11 hours to complete", "Flexible schedule", "3,110 reviews", "",
     "['Docker', 'Kubernetes', 'DevOps']", "Intro", "Rav Ahuja", "IBM", "k8s",
     "https://www.coursera.org/learn/docker-kubernetes"],
    # Slug absent from the spine: must NOT create a course row.
    ["Fashion as Design", "4.8", "Beginner level", "20 hours (approximately)",
     "Flexible schedule", "2,813 reviews", "", "['Art History', 'Art']", "Intro",
     "Anna Burckhardt", "The Museum of Modern Art", "art",
     "https://www.coursera.org/learn/fashion-design"],
]


def _write(path: Path, header, rows, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decoded = [[c.encode("latin-1", "ignore").decode("utf-8", "replace")
                if isinstance(c, str) and "\\x" not in repr(c)[:2] else c
                for c in row] for row in rows]
    with open(path, "w", encoding=encoding, newline="", errors="replace") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(decoded)


def build(root: Path) -> None:
    raw = root / "data" / "raw"
    # coursera2021 written as cp1252 to reproduce the real encoding problem
    _write(raw / "coursera2021" / "Coursera.csv",
           COURSERA2021_HEADER, COURSERA2021_ROWS, "cp1252")
    _write(raw / "coursera2025" / "Coursera.csv",
           COURSERA2025_HEADER, COURSERA2025_ROWS, "utf-8")
    _write(raw / "edx" / "edx_courses.csv", EDX_HEADER, EDX_ROWS, "utf-8")
    _write(raw / "coursera_hours" / "CourseraDataset-Unclean.csv",
           COURSERA_HOURS_HEADER, COURSERA_HOURS_ROWS, "utf-8")
    print(f"[fixtures] wrote synthetic sources under {raw}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic source CSVs")
    ap.add_argument("--root", required=True, help="scratch root (THESIS_DATA_ROOT)")
    build(Path(ap.parse_args().root))


if __name__ == "__main__":
    main()
