"""
tests/unit/test_step3.py

Unit tests for Step 3: AWSSession, AWSOrganizations, AWSProvider.
Pre-registers boto3/botocore mocks before any imports.
"""

from __future__ import annotations
import sys, os
from unittest.mock import MagicMock

# ---- Mock boto3/botocore BEFORE any imports ----
class FakeClientError(Exception):
    def __init__(self, error_response, operation_name="Op"):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(str(error_response))

class FakeNoCredentialsError(Exception): pass
class FakeProfileNotFound(Exception): pass

_botocore_exc = MagicMock()
_botocore_exc.ClientError = FakeClientError
_botocore_exc.NoCredentialsError = FakeNoCredentialsError
_botocore_exc.ProfileNotFound = FakeProfileNotFound

_botocore = MagicMock()
_botocore.exceptions = _botocore_exc
_boto3 = MagicMock()

sys.modules.setdefault("boto3", _boto3)
sys.modules.setdefault("botocore", _botocore)
sys.modules.setdefault("botocore.exceptions", _botocore_exc)
# ------------------------------------------------

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from providers.aws.session import AWSSession, AWSSessionError, AssumeRoleError, CachedCredentials
from providers.aws.organizations import AWSOrganizations, OrgAccount, OrganizationInfo, OrganizationsError
from providers.aws.provider import AWSProvider
from core.base_provider import AccountScanError


def make_expiry(secs=3600):
    return datetime.now(timezone.utc) + timedelta(seconds=secs)


def make_mock_boto3_session(account_id="123456789012"):
    session = MagicMock()
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": account_id, "Arn": f"arn:aws:iam::{account_id}:user/scanner", "UserId": "AIDA"}
    sts.assume_role.return_value = {"Credentials": {"AccessKeyId": "AK", "SecretAccessKey": "SK", "SessionToken": "ST", "Expiration": make_expiry(3600)}}
    iam = MagicMock()
    iam.list_account_aliases.return_value = {"AccountAliases": ["my-prod-alias"]}
    ec2 = MagicMock()
    ec2.describe_regions.return_value = {"Regions": [{"RegionName": "us-east-1"}, {"RegionName": "us-west-2"}, {"RegionName": "eu-west-1"}]}
    clients = {"sts": sts, "iam": iam, "ec2": ec2}
    session.client.side_effect = lambda svc, **kw: clients.get(svc, MagicMock())
    return session


def make_mock_orgs_client(accounts=None):
    if accounts is None:
        accounts = [
            {"Id": "111111111111", "Name": "Production", "Email": "p@e.com", "Status": "ACTIVE", "JoinedMethod": "CREATED"},
            {"Id": "222222222222", "Name": "Staging", "Email": "s@e.com", "Status": "ACTIVE", "JoinedMethod": "INVITED"},
            {"Id": "333333333333", "Name": "OldAccount", "Email": "o@e.com", "Status": "SUSPENDED", "JoinedMethod": "INVITED"},
        ]
    client = MagicMock()
    client.describe_organization.return_value = {"Organization": {"Id": "o-example123", "MasterAccountId": "000000000000", "MasterAccountEmail": "m@e.com", "FeatureSet": "ALL"}}
    client.list_roots.return_value = {"Roots": [{"Id": "r-root", "Name": "Root"}]}
    pagers = {
        "list_accounts": (lambda a=accounts: _pager([{"Accounts": a}]))(),
        "list_accounts_for_parent": _pager([{"Accounts": []}]),
        "list_organizational_units_for_parent": _pager([{"OrganizationalUnits": []}]),
        "list_policies": _pager([{"Policies": [{"Id": "p-001", "Name": "DenyAllSCP", "Description": "", "AwsManaged": False}]}]),
    }
    client.get_paginator.side_effect = lambda op: pagers.get(op, MagicMock())
    return client


def _pager(pages):
    p = MagicMock()
    p.paginate.return_value = pages
    return p


def _init_aws_session(config=None, account_id="123456789012"):
    mock_base = make_mock_boto3_session(account_id=account_id)
    aws_session = AWSSession(config=config or {})
    aws_session._base_session = mock_base
    aws_session._identity = {"Account": account_id, "Arn": f"arn:aws:iam::{account_id}:user/s", "UserId": "X"}
    aws_session._home_account_id = account_id
    return aws_session, mock_base


def _make_provider(config=None, account_id="123456789012"):
    provider = AWSProvider(config=config or {})
    aws_session, mock_base = _init_aws_session(account_id=account_id)
    provider._aws_session = aws_session
    provider._authenticated = True
    return provider, mock_base


# ===========================================================================
class TestCachedCredentials(unittest.TestCase):
    def test_not_expired(self):
        c = CachedCredentials("AK", "SK", "ST", make_expiry(3600))
        self.assertFalse(c.is_expired)

    def test_expired(self):
        c = CachedCredentials("AK", "SK", "ST", datetime.now(timezone.utc) - timedelta(seconds=1))
        self.assertTrue(c.is_expired)

    def test_needs_refresh_when_close(self):
        c = CachedCredentials("AK", "SK", "ST", make_expiry(200))  # < 300s buffer
        self.assertTrue(c.needs_refresh)

    def test_no_refresh_needed_with_time_remaining(self):
        c = CachedCredentials("AK", "SK", "ST", make_expiry(3600))
        self.assertFalse(c.needs_refresh)

    def test_boto3_credentials_keys(self):
        c = CachedCredentials("AKID", "SECRET", "TOKEN", make_expiry())
        b = c.as_boto3_credentials
        self.assertEqual(b["aws_access_key_id"], "AKID")
        self.assertEqual(b["aws_secret_access_key"], "SECRET")
        self.assertEqual(b["aws_session_token"], "TOKEN")


# ===========================================================================
class TestAWSSession(unittest.TestCase):
    def test_home_account_stored(self):
        s, _ = _init_aws_session(account_id="123456789012")
        self.assertEqual(s.home_account_id, "123456789012")

    def test_home_account_returns_base_session(self):
        s, mock_base = _init_aws_session(account_id="123456789012")
        self.assertIs(s.get_session_for_account("123456789012"), mock_base)

    def test_cross_account_calls_assume_role(self):
        s, mock_base = _init_aws_session(account_id="000000000000", config={"role_name": "SecurityAudit"})
        s._session_from_credentials = lambda c: MagicMock()
        s.get_session_for_account("111111111111")
        sts = mock_base.client("sts")
        sts.assume_role.assert_called_once()
        self.assertIn("111111111111", sts.assume_role.call_args[1]["RoleArn"])
        self.assertIn("SecurityAudit", sts.assume_role.call_args[1]["RoleArn"])

    def test_credentials_cached_second_call_no_assume(self):
        s, mock_base = _init_aws_session(account_id="000000000000")
        s._session_from_credentials = lambda c: MagicMock()
        s.get_session_for_account("111111111111")
        s.get_session_for_account("111111111111")
        self.assertEqual(mock_base.client("sts").assume_role.call_count, 1)

    def test_external_id_in_assume_role(self):
        s, mock_base = _init_aws_session(account_id="000000000000", config={"external_id": "ext-xyz"})
        s._session_from_credentials = lambda c: MagicMock()
        s.get_session_for_account("111111111111")
        self.assertEqual(mock_base.client("sts").assume_role.call_args[1].get("ExternalId"), "ext-xyz")

    def test_assume_role_failure_raises_error(self):
        s, mock_base = _init_aws_session(account_id="000000000000")
        mock_base.client("sts").assume_role.side_effect = FakeClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        with self.assertRaises(AssumeRoleError) as ctx:
            s.get_session_for_account("999999999999")
        self.assertIn("999999999999", str(ctx.exception))

    def test_invalidate_specific_account(self):
        s, _ = _init_aws_session()
        c = CachedCredentials("AK", "SK", "ST", make_expiry())
        s._credential_cache["111"] = c; s._credential_cache["222"] = c
        s.invalidate_cache("111")
        self.assertNotIn("111", s._credential_cache)
        self.assertIn("222", s._credential_cache)

    def test_invalidate_all(self):
        s, _ = _init_aws_session()
        s._credential_cache["111"] = CachedCredentials("AK", "SK", "ST", make_expiry())
        s._credential_cache["222"] = CachedCredentials("AK", "SK", "ST", make_expiry())
        s.invalidate_cache()
        self.assertEqual(len(s._credential_cache), 0)

    def test_get_account_alias(self):
        s, _ = _init_aws_session()
        self.assertEqual(s.get_account_alias(), "my-prod-alias")

    def test_get_enabled_regions(self):
        s, _ = _init_aws_session()
        regions = s.get_enabled_regions()
        self.assertIn("us-east-1", regions)
        self.assertIn("eu-west-1", regions)

    def test_repr_before_init(self):
        s = AWSSession({})
        self.assertIn("initialised=False", repr(s))

    def test_repr_after_init(self):
        s, _ = _init_aws_session(account_id="123456789012")
        self.assertIn("123456789012", repr(s))

    def test_assumed_accounts_tracked(self):
        s, _ = _init_aws_session(account_id="000000000000")
        s._session_from_credentials = lambda c: MagicMock()
        s.get_session_for_account("111111111111")
        s.get_session_for_account("222222222222")
        self.assertIn("111111111111", s.assumed_accounts)
        self.assertIn("222222222222", s.assumed_accounts)


# ===========================================================================
class TestAssumeRoleError(unittest.TestCase):
    def test_access_denied_is_helpful(self):
        e = AssumeRoleError("111", "SecurityAudit", "AccessDenied", "Denied")
        self.assertIn("trust policy", str(e))
        self.assertIn("111", str(e))

    def test_no_such_entity_mentions_creation(self):
        e = AssumeRoleError("111", "SecurityAudit", "NoSuchEntity", "Not found")
        self.assertIn("does not exist", str(e))

    def test_unknown_code_graceful(self):
        e = AssumeRoleError("111", "SecurityAudit", "WeirdCode", "Mystery")
        self.assertIn("111", str(e))


# ===========================================================================
class TestAWSOrganizations(unittest.TestCase):
    def _orgs(self, client=None):
        o = AWSOrganizations(session=MagicMock())
        o._client = client or make_mock_orgs_client()
        return o

    def test_is_enabled_true(self):
        self.assertTrue(self._orgs().is_enabled())

    def test_is_enabled_false_on_exception(self):
        c = MagicMock()
        c.describe_organization.side_effect = FakeClientError(
            {"Error": {"Code": "AWSOrganizationsNotInUseException", "Message": "N/A"}})
        self.assertFalse(self._orgs(c).is_enabled())

    def test_describe_returns_org_info(self):
        info = self._orgs().describe()
        self.assertIsInstance(info, OrganizationInfo)
        self.assertEqual(info.org_id, "o-example123")
        self.assertTrue(info.is_all_features)

    def test_describe_cached(self):
        o = self._orgs()
        o.describe(); o.describe()
        o._client.describe_organization.assert_called_once()

    def test_list_accounts_active_only_default(self):
        accounts = self._orgs().list_accounts()
        self.assertEqual(len(accounts), 2)
        self.assertNotIn("333333333333", accounts)

    def test_list_accounts_with_suspended(self):
        accounts = self._orgs().list_accounts(include_suspended=True)
        self.assertEqual(len(accounts), 3)
        self.assertIn("333333333333", accounts)

    def test_account_objects_fields(self):
        objs = self._orgs().list_account_objects()
        prod = next(a for a in objs if a.account_id == "111111111111")
        self.assertEqual(prod.name, "Production")
        self.assertTrue(prod.is_active)

    def test_account_objects_cached(self):
        o = self._orgs()
        o.list_account_objects(); o.list_account_objects()
        calls = [c for c in o._client.get_paginator.call_args_list if c[0][0] == "list_accounts"]
        self.assertEqual(len(calls), 1)

    def test_get_account(self):
        a = self._orgs().get_account("111111111111")
        self.assertIsNotNone(a)
        self.assertEqual(a.name, "Production")

    def test_get_account_unknown(self):
        self.assertIsNone(self._orgs().get_account("999999999999"))

    def test_get_account_name(self):
        self.assertEqual(self._orgs().get_account_name("111111111111"), "Production")

    def test_get_account_name_fallback(self):
        self.assertEqual(self._orgs().get_account_name("999"), "999")

    def test_scps_returned(self):
        scps = self._orgs().list_scps()
        self.assertEqual(len(scps), 1)
        self.assertEqual(scps[0].name, "DenyAllSCP")

    def test_scps_empty_billing_only(self):
        c = make_mock_orgs_client()
        c.describe_organization.return_value = {"Organization": {"Id": "o-1", "MasterAccountId": "0", "MasterAccountEmail": "m@e.com", "FeatureSet": "CONSOLIDATED_BILLING"}}
        self.assertEqual(self._orgs(c).list_scps(), [])

    def test_access_denied_raises_orgs_error(self):
        c = make_mock_orgs_client()
        bad = MagicMock(); bad.paginate.side_effect = FakeClientError({"Error": {"Code": "AccessDeniedException", "Message": "Denied"}})
        c.get_paginator.side_effect = lambda op: bad if op == "list_accounts" else MagicMock()
        with self.assertRaises(OrganizationsError) as ctx:
            self._orgs(c).list_account_objects()
        self.assertIn("management account", str(ctx.exception))

    def test_get_summary(self):
        s = self._orgs().get_summary()
        self.assertTrue(s["enabled"])
        self.assertIn("org_info", s)

    def test_org_account_display_name(self):
        a = OrgAccount("123", "Prod", "p@e.com", "ACTIVE", "CREATED")
        self.assertEqual(a.display_name, "Prod (123)")

    def test_org_account_active_status(self):
        self.assertTrue(OrgAccount("1", "A", "a@a.com", "ACTIVE", "CREATED").is_active)
        self.assertFalse(OrgAccount("2", "B", "b@b.com", "SUSPENDED", "CREATED").is_active)

    def test_org_info_features(self):
        self.assertTrue(OrganizationInfo("o-1", "0", "m@e.com", "ALL").is_all_features)
        self.assertFalse(OrganizationInfo("o-2", "0", "m@e.com", "CONSOLIDATED_BILLING").is_all_features)


# ===========================================================================
class TestAWSProviderStep3(unittest.TestCase):
    def test_validate_config_bad_regions(self):
        errors = AWSProvider(config={"regions": "string-not-list"}).validate_config()
        self.assertTrue(len(errors) > 0)

    def test_validate_config_bad_duration(self):
        errors = AWSProvider(config={"session_duration": 100}).validate_config()
        self.assertTrue(len(errors) > 0)

    def test_validate_config_valid(self):
        errors = AWSProvider(config={"regions": ["us-east-1"], "session_duration": 3600}).validate_config()
        self.assertEqual(errors, [])

    def test_single_account_scan(self):
        p, _ = _make_provider(account_id="123456789012")
        self.assertEqual(p.get_accounts(), ["123456789012"])

    def test_explicit_account_list(self):
        p, _ = _make_provider(config={"account_ids": ["111", "222"]})
        self.assertEqual(p.get_accounts(), ["111", "222"])

    def test_org_scan_uses_organizations(self):
        p, _ = _make_provider(config={"scan_org": True})
        mock_orgs = MagicMock()
        mock_orgs.is_enabled.return_value = True
        mock_orgs.list_accounts.return_value = ["111111111111", "222222222222"]
        mock_orgs.list_account_objects.return_value = [
            OrgAccount("111111111111", "Production", "p@e.com", "ACTIVE", "CREATED"),
        ]
        p._organizations = mock_orgs
        accounts = p.get_accounts()
        self.assertIn("111111111111", accounts)

    def test_org_scan_fallback_when_disabled(self):
        p, _ = _make_provider(config={"scan_org": True}, account_id="123")
        mock_orgs = MagicMock(); mock_orgs.is_enabled.return_value = False
        p._organizations = mock_orgs
        self.assertEqual(p.get_accounts(), ["123"])

    def test_get_account_name_from_cache(self):
        p, _ = _make_provider()
        p._account_names["111"] = "Production"
        self.assertEqual(p.get_account_name("111"), "Production")

    def test_get_account_name_from_org(self):
        p, _ = _make_provider()
        mock_orgs = MagicMock(); mock_orgs.get_account_name.return_value = "Staging"
        p._organizations = mock_orgs
        self.assertEqual(p.get_account_name("222"), "Staging")

    def test_get_account_name_iam_alias_for_home(self):
        p, _ = _make_provider(account_id="123456789012")
        self.assertEqual(p.get_account_name("123456789012"), "my-prod-alias")

    def test_get_account_name_fallback(self):
        p, _ = _make_provider()
        self.assertEqual(p.get_account_name("999999999999"), "999999999999")

    def test_get_regions_from_config(self):
        p, _ = _make_provider(config={"regions": ["eu-west-1"]})
        self.assertEqual(p.get_regions(), ["eu-west-1"])

    def test_get_regions_enumerates(self):
        p, _ = _make_provider()
        regions = p.get_regions()
        self.assertIn("us-east-1", regions)

    def test_pre_scan_skips_home_account(self):
        p, mock_base = _make_provider(account_id="123456789012")
        p.pre_scan_hook("123456789012")
        mock_base.client("sts").assume_role.assert_not_called()

    def test_pre_scan_assumes_role_for_member(self):
        p, mock_base = _make_provider(account_id="000000000000")
        p._aws_session._session_from_credentials = lambda c: MagicMock()
        p.pre_scan_hook("111111111111")
        mock_base.client("sts").assume_role.assert_called_once()
        self.assertIn("111111111111", mock_base.client("sts").assume_role.call_args[1]["RoleArn"])

    def test_pre_scan_hook_raises_account_scan_error(self):
        p, mock_base = _make_provider(account_id="000000000000")
        mock_base.client("sts").assume_role.side_effect = FakeClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}})
        with self.assertRaises(AccountScanError):
            p.pre_scan_hook("111111111111")

    def test_repr_after_auth(self):
        p, _ = _make_provider(account_id="123456789012")
        r = repr(p)
        self.assertIn("123456789012", r)
        self.assertIn("True", r)

    def test_supports_multi_account_org(self):
        self.assertTrue(AWSProvider(config={"scan_org": True}).supports_multi_account())

    def test_supports_multi_account_explicit(self):
        self.assertTrue(AWSProvider(config={"account_ids": ["111"]}).supports_multi_account())

    def test_not_multi_account_single(self):
        self.assertFalse(AWSProvider(config={}).supports_multi_account())


if __name__ == "__main__":
    unittest.main(verbosity=2)