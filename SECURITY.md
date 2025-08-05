# Security Policy

## Supported Versions

We actively support the following versions of MonStim Analyzer with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take the security of MonStim Analyzer seriously. If you discover a security vulnerability, please follow these steps:

### For Security Issues

**Do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please:

1. **Email us directly:** Send details to aeworthy@emory.edu
2. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes (if available)
   - Your contact information for follow-up

### What to Expect

- **Acknowledgment:** We'll acknowledge receipt within 48 hours
- **Assessment:** We'll assess the vulnerability within 5 business days
- **Communication:** We'll keep you informed of our progress
- **Resolution:** We'll work to resolve critical issues as quickly as possible
- **Credit:** We'll credit you in the security advisory (if desired)

### Security Considerations

MonStim Analyzer handles sensitive research data, so we're particularly concerned about:

- **Data privacy:** Unauthorized access to EMG research data
- **Code execution:** Malicious code execution through data files
- **File system access:** Improper file system permissions or access
- **Network security:** Any network-related vulnerabilities
- **Dependency vulnerabilities:** Issues in third-party libraries

### Common Vulnerability Types

Please report any issues related to:

- **Arbitrary code execution** through malicious CSV files
- **Path traversal** vulnerabilities in file operations
- **Injection attacks** through user input
- **Privilege escalation** on the host system
- **Data leakage** through log files or temporary files
- **Denial of service** through malformed data files

### Responsible Disclosure

We follow responsible disclosure practices:

- We'll work with you to understand and resolve the issue
- We'll coordinate the timing of public disclosure
- We'll provide security advisories for significant vulnerabilities
- We'll acknowledge your contribution (with your permission)

### Security Best Practices for Users

While using MonStim Analyzer:

- **Download only from official sources:** GitHub releases or trusted repositories
- **Verify file integrity:** Check checksums when available
- **Use appropriate permissions:** Don't run with unnecessary privileges
- **Keep software updated:** Install security updates promptly
- **Protect your data:** Use appropriate file permissions for sensitive data
- **Scan unknown files:** Be cautious with data files from untrusted sources

### Security Update Process

When security issues are identified:

1. **Assessment:** We evaluate the severity and impact
2. **Fix development:** We develop and test security patches
3. **Testing:** We conduct thorough testing of security fixes
4. **Release:** We release security updates as patch versions
5. **Communication:** We notify users through multiple channels
6. **Documentation:** We update security documentation as needed

### Security Resources

- **Security Advisories:** [GitHub Security Advisories](https://github.com/AEWorthy/MonStim-Analyzer/security/advisories)
- **Dependency Updates:** We monitor dependencies for known vulnerabilities
- **Security Guidelines:** Follow the security best practices in our documentation

### Legal Considerations

- Research data may be subject to institutional policies
- Some EMG data may be considered sensitive or confidential
- Users are responsible for compliance with applicable data protection laws
- This software is provided "as is" without warranty (see LICENSE)

---

Thank you for helping keep MonStim Analyzer and our research community secure! 