import os
import pooch
import pandas as pd
from typing import Optional

from relbench.base import Database, Dataset, Table
from relbench.utils import clean_datetime, unzip_processor


class StackDataset(Dataset):
    """
    For stack dataset, there is an augmentation.
    We split the tags in "posts" table as a new table "tags".
    And add a new relationship table post-tags to represent the many-to-many relationship
    """

    # 3 months gap
    val_timestamp = pd.Timestamp("2020-10-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        # remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes",
                     "DownVotes", "LastAccessDate"],
            inplace=True,
        )

        posts.drop(
            columns=[
                "ViewCount",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "CommunityOwnedDate",
                "ClosedDate",
                "LastEditDate",
                "LastActivityDate",
                # "Score",
                "LastEditorDisplayName",
                "LastEditorUserId",
            ],
            inplace=True,
        )

        # comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        comments = clean_datetime(comments, "CreationDate")
        badges = clean_datetime(badges, "Date")
        postLinks = clean_datetime(postLinks, "CreationDate")
        postHistory = clean_datetime(postHistory, "CreationDate")
        votes = clean_datetime(votes, "CreationDate")
        users = clean_datetime(users, "CreationDate")
        posts = clean_datetime(posts, "CreationDate")

        # add an additional table "tags"
        # add an additional relationship table "post-tags"

        posts['TagList'] = posts['Tags'].str.findall(r'<(.*?)>')
        # str-> list  <bayesian><prior><elicitation> -> ['bayesian', 'prior', 'elicitation']
        post_tag = posts[['Id', 'TagList']].explode(
            'TagList').rename(columns={'TagList': 'TagName'})
        post_tag = post_tag.dropna(subset=['TagName']).reset_index(drop=True)

        tags = pd.DataFrame(post_tag['TagName'].unique(), columns=['TagName'])
        tags['TagId'] = range(1, len(tags) + 1)
        post_tag = post_tag.merge(tags, on='TagName', how='left')[
            ['Id', 'TagId']]

        # clear the schema name
        post_tag['PostId'] = post_tag['Id']
        post_tag['Id'] = range(1, len(post_tag) + 1)
        tags['Id'] = tags['TagId']
        tags.drop(columns=['TagId'], inplace=True)

        # drop 'Tags' column in posts
        posts.drop(columns=['Tags'], inplace=True)
        posts.drop(columns=['TagList'], inplace=True)

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            fkey_col_to_pkey_table={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkey_col_to_pkey_table={
                "UserId": "users",
            },
            pkey_col="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "RelatedPostId": "posts",  # is this allowed? two foreign keys into the same primary
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "ParentId": "posts",  # notice the self-reference
                "AcceptedAnswerId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        # add the new tables
        tables["tags"] = Table(
            df=pd.DataFrame(tags),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col=None,
        )

        # add the new relationship table
        tables["postTag"] = Table(
            df=pd.DataFrame(post_tag),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "TagId": "tags",
            },
            pkey_col="Id",
            time_col=None,
        )

        return Database(tables)


# ============================================================================
# Self-registration with DatabaseFactory
# ============================================================================

def _register_stack():
    """Register Stack dataset and tasks with DatabaseFactory."""
    from .database_factory import DatabaseFactory
    from relbench.tasks import stack

    def _load_stack_dataset(cache_dir: Optional[str] = None) -> Dataset:
        """Load the Stack dataset."""
        cache_root_dir = os.path.join("~", ".cache", "relbench")
        cache_root_dir = os.path.expanduser(cache_root_dir)
        cache_dir = cache_dir if cache_dir else os.path.join(
            cache_root_dir, "stack")
        # print("Stack dataset cache dir:", cache_dir)
        return StackDataset(cache_dir=cache_dir)

    # Register dataset
    DatabaseFactory.register_dataset("stack", _load_stack_dataset)

    # Register tasks
    DatabaseFactory.register_task(
        "stack", "user-engagement", stack.UserEngagementTask)
    DatabaseFactory.register_task("stack", "user-badge", stack.UserBadgeTask)
    DatabaseFactory.register_task("stack", "post-vote", stack.PostVotesTask)


# Auto-register when this module is imported
_register_stack()
