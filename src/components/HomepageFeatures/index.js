"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = HomepageFeatures;
var clsx_1 = require("clsx");
var Heading_1 = require("@theme/Heading");
var styles_module_css_1 = require("./styles.module.css");
var FeatureList = [
    {
        title: 'Structured Learning Path',
        Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
        description: (<>
        Step-by-step notes covering Python, Data Structures, Machine Learning,
        Deep Learning, PyTorch, FastAPI, and MLOps — from fundamentals to
        industry-ready practices.
      </>),
    },
    {
        title: 'Code-First Notes',
        Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
        description: (<>
        Every concept is backed by working code examples, visual explanations,
        and common interview questions so you understand not just <em>how</em>,
        but <em>why</em>.
      </>),
    },
    {
        title: 'Real-World Projects',
        Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
        description: (<>
        Build CNNs, RNNs, LSTMs, transfer learning pipelines, REST APIs with
        FastAPI, and end-to-end MLOps systems — the same stack used in
        production AI teams.
      </>),
    },
];
function Feature(_a) {
    var title = _a.title, Svg = _a.Svg, description = _a.description;
    return (<div className={(0, clsx_1.default)('col col--4')}>
      <div className="text--center">
        <Svg className={styles_module_css_1.default.featureSvg} role="img"/>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading_1.default as="h3">{title}</Heading_1.default>
        <p>{description}</p>
      </div>
    </div>);
}
function HomepageFeatures() {
    return (<section className={styles_module_css_1.default.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map(function (props, idx) { return (<Feature key={idx} {...props}/>); })}
        </div>
      </div>
    </section>);
}
